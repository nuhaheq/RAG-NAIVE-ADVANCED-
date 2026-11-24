import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:
    st.error("The 'groq' package is not installed. Check requirements.txt.")
    st.stop()

from sentence_transformers import CrossEncoder, SentenceTransformer

st.set_page_config(page_title="AFib Clinical Decision Support", layout="wide")
st.title("Retrieval Augmented Generation-Based Clinical Decision Support System for Anticoagulation in Atrial Fibrillation")

# ---------------------------------------------------------------------------
# SETTINGS (report these in Methods 3.3-3.6)
# ---------------------------------------------------------------------------
INDEX_DIR = "index_mpnet_v2"           # folder in this repo, copied from Google Drive (built by build_kb_index_colab.ipynb)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ONE fixed model for every answer (no fallback to another model). llama-3.3-70b-versatile was shut down by Groq
# on 16 Aug 2026; openai/gpt-oss-120b is Groq's recommended replacement. Override with GROQ_MODEL in Streamlit secrets.
GROQ_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = "medium"            # gpt-oss models only; reasoning is kept out of the answer text
TEMPERATURE = 0.0                      # repeatable answers
MAX_TOKENS = 8192                      # room for the model's hidden reasoning + the full answer
DENSE_TOP_N = 20       # dense candidates entering fusion
BM25_TOP_N = 20        # BM25 candidates entering fusion
RRF_K = 60             # Reciprocal Rank Fusion constant (Cormack et al., 2009)
RERANK_POOL_SIZE = 20  # fused candidates sent to the cross-encoder
TOP_N_CONTEXT = 5      # chunks given to the LLM (both arms)

# The index can be uploaded either as the folder itself or as index_mpnet_v2.zip (extracted on first run).
if not os.path.isdir(INDEX_DIR) and os.path.exists(f"{INDEX_DIR}.zip"):
    import zipfile
    with zipfile.ZipFile(f"{INDEX_DIR}.zip") as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
        if all(n.startswith(f"{INDEX_DIR}/") for n in names):
            z.extractall(".", members=names)                 # zip contains the folder
        else:
            os.makedirs(INDEX_DIR, exist_ok=True)              # zip contains the files directly
            for n in names:
                with z.open(n) as src, open(os.path.join(INDEX_DIR, os.path.basename(n)), "wb") as dst:
                    dst.write(src.read())
if not os.path.exists(os.path.join(INDEX_DIR, "kb_utils.py")):
    st.error(f"'{INDEX_DIR}/' (or {INDEX_DIR}.zip) with kb_utils.py was not found in the repo.")
    st.stop()

sys.path.insert(0, INDEX_DIR)
import kb_utils  # noqa: E402  (same tokenizer + search code used to build the index)

if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

# ---------------------------------------------------------------------------
# SIDEBAR
#   Naive    : dense (FAISS, cosine) top-5
#   Advanced : dense + BM25 fused with RRF -> cross-encoder rerank, top-5
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ RAG Architecture Settings")
MODE_NAIVE = "Naive RAG (dense only)"
MODE_ADVANCED = "Advanced RAG (hybrid + cross-encoder reranking)"
ARM_OF = {MODE_NAIVE: "naive", MODE_ADVANCED: "advanced"}
rag_mode = st.sidebar.radio("Select retrieval mode:", options=[MODE_NAIVE, MODE_ADVANCED], index=1)
st.sidebar.caption(f"Model: `{GROQ_MODEL}` · temperature {TEMPERATURE} · top {TOP_N_CONTEXT} chunks")


@st.cache_data(ttl=3600, show_spinner=False)
def groq_models():
    return sorted(m.id for m in Groq(api_key=st.secrets["GROQ_API_KEY"]).models.list().data)


try:
    _available = groq_models()
except Exception as e:
    st.error(f"Could not reach Groq to check the model list: {e}")
    st.stop()
if GROQ_MODEL not in _available:
    st.error(f"Groq model `{GROQ_MODEL}` is not available on your account. Available models: " + ", ".join(f"`{m}`" for m in _available)
             + ". Add GROQ_MODEL = \"<one of these>\" to your Streamlit secrets and reboot.")
    st.stop()


@st.cache_resource
def load_everything():
    chunks, index, bm25 = kb_utils.load_kb(INDEX_DIR)
    if not (len(chunks) == index.ntotal == bm25.corpus_size):
        raise RuntimeError("Index files do not match each other - re-copy the whole index_mpnet_v2 folder.")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    reranker = CrossEncoder(RERANKER, device="cpu")
    manifest = json.load(open(f"{INDEX_DIR}/index_manifest.json")) if os.path.exists(f"{INDEX_DIR}/index_manifest.json") else {}
    return chunks, index, bm25, embedder, reranker, manifest


try:
    chunks, faiss_index, bm25, embedder, reranker, manifest = load_everything()
except Exception as e:
    st.error(f"Could not load the knowledge base from '{INDEX_DIR}/': {e}")
    st.stop()
st.sidebar.caption(f"Knowledge base: {len(chunks)} chunks · built {manifest.get('built', '?')}")


def retrieve(question: str, arm: str):
    """Returns the row numbers of the chunks given to the LLM, plus rank diagnostics."""
    dense = kb_utils.dense_search(question, embedder, faiss_index, k=max(DENSE_TOP_N, TOP_N_CONTEXT))
    dense_rank = {r: i for i, (r, _) in enumerate(dense, start=1)}
    if arm == "naive":
        return [r for r, _ in dense[:TOP_N_CONTEXT]], {"dense_rank": dense_rank}
    sparse = [(r, s) for r, s in kb_utils.bm25_search(question, bm25, k=BM25_TOP_N) if s > 0]
    bm25_rank = {r: i for i, (r, _) in enumerate(sparse, start=1)}
    fused = [r for r, _ in kb_utils.rrf(dense[:DENSE_TOP_N], sparse, k_rrf=RRF_K, top_n=RERANK_POOL_SIZE)]
    scores = reranker.predict([(question, chunks[r]["text"]) for r in fused])
    order = np.argsort(-np.asarray(scores))[:TOP_N_CONTEXT]
    return [fused[i] for i in order], {"dense_rank": dense_rank, "bm25_rank": bm25_rank,
                                       "rerank_score": {fused[i]: float(scores[i]) for i in order}}


def body(c: dict) -> str:
    return c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]


def build_prompt(context_text: str, user_query: str) -> str:
    return PROMPT_TEMPLATE.replace("{context_text}", context_text).replace("{user_query}", user_query)


PROMPT_TEMPLATE = """
You are a Board-Certified Clinical Specialist Professor in Atrial Fibrillation (AFib) Pharmacotherapy.

YOUR OBJECTIVE:
Provide structured, highly accurate clinical recommendations in response to the user's query using ONLY the provided official guideline context chunks. You must attribute every clinical statement to its exact source guideline and section header.

==================================================
STRICT CLINICAL SAFETY RULES:
==================================================
1. STRICT GROUNDING: Base every recommendation solely on the CONTEXT DATA below. Do NOT use outside medical knowledge or unvalidated general assumptions.
2. ABSENCE OF EVIDENCE: If the provided CONTEXT DATA does not contain sufficient information to safely answer the user's query, explicitly state: "The retrieved guideline context does not contain sufficient clinical guidance to answer this specific query."
3. MANDATORY IN-LINE CITATIONS: Every clinical recommendation, dose, risk-score threshold, or contraindication MUST be followed immediately by an inline bracketed citation containing the Guideline ID and Section Name.
   - Format: [Guideline_ID | Section_Header]
   - Example: "Apixaban 5 mg twice daily is recommended for stroke prevention [ACC_AHA_2023 | 4.1 Stroke Prevention]."
4. NO UNVALIDATED RATIONALES: Only cite active, validated recommendations.
5. CONTRADICTION HANDLING: If multiple guidelines give conflicting recommendations, explicitly compare both views clearly.

==================================================
CONTEXT DATA FROM GUIDELINES:
==================================================
{context_text}

==================================================
RESPONSE STRUCTURE:
==================================================
Structure your response using the following format:

**Clinical Recommendation Summary**
- Provide a direct 2-3 sentence summary answering the query with inline citations.

**Detailed Guideline Guidance**
- Break down specific dosing, risk stratification scores, renal adjustments, or therapeutic choices using bullet points.
- Ensure EVERY bullet point has its corresponding [Guideline_ID | Section_Header] citation.

**Guideline Provenance Matrix**
Provide a Markdown table summarizing all guidelines cited:

| Guideline / Source | Section Referenced | Key Recommendation Summary |
| :--- | :--- | :--- |

==================================================
USER QUERY: {user_query}

ANSWER:
"""


def generate(prompt: str, max_tries: int = 6):
    """One fixed model. Only rate limits / server errors are retried; any other error stops at once."""
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    extra = {"reasoning_effort": REASONING_EFFORT} if GROQ_MODEL.startswith("openai/gpt-oss") else {}
    last = None
    for attempt in range(max_tries):
        try:
            r = client.chat.completions.create(model=GROQ_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                               messages=[{"role": "user", "content": prompt}], **extra)
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError(f"empty answer (finish_reason={r.choices[0].finish_reason}) - raise MAX_TOKENS")
            return text, r.model
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is not None and status != 429 and status < 500:
                raise                                   # e.g. 404 model not found, 401 bad key: retrying will not help
            last = e
            time.sleep(min(30, 5 * 2 ** attempt))
    raise RuntimeError(f"Groq failed after {max_tries} tries: {last}")


def answer_one(case_id: str, question: str, arm: str, reference: str = "") -> dict:
    rows, diag = retrieve(question, arm)
    context_text = "\n".join(f"- [{chunks[r]['guideline_id']} | {chunks[r]['section']}] {body(chunks[r])}" for r in rows)
    answer, model_id = generate(build_prompt(context_text, question))
    return {
        "case_id": case_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rag_mode": arm, "model_id": model_id, "temperature": TEMPERATURE,
        "query": question, "answer": answer,
        "retrieved_contexts": json.dumps([chunks[r]["text"] for r in rows], ensure_ascii=False),
        "retrieved_contexts_count": len(rows),
        "chunk_ids": json.dumps([chunks[r]["chunk_id"] for r in rows]),
        "diagnostics": json.dumps({k: {chunks[r]["chunk_id"]: v for r, v in d.items() if r in rows} for k, d in diag.items()}),
        "ground_truth": reference,
    }


tab_one, tab_batch = st.tabs(["🩺 Ask one question", "🧪 Evaluation run (all test questions)"])

# ---------------------------------------------------------------------------
# TAB 1 - single question (demo)
# ---------------------------------------------------------------------------
with tab_one:
    case_id = st.text_input("CASE ID (optional, e.g. Q01):", value="")
    user_query = st.text_area("INSERT CLINICAL QUERY:",
                              placeholder="e.g., What is the recommended apixaban dose in an 82-year-old weighing 58 kg?")
    if st.button("Generate Answer", type="primary"):
        if not user_query.strip():
            st.warning("PLEASE INSERT CLINICAL QUERY FIRST.")
        else:
            with st.spinner("SEARCHING GUIDELINES AND GENERATING ANSWER..."):
                try:
                    rec = answer_one(case_id.strip(), user_query, ARM_OF[rag_mode])
                    st.caption(f"🤖 `{rec['model_id']}` | RAG mode: `{rag_mode}`")
                    st.markdown(rec["answer"])
                    with st.expander("🔍 Retrieved chunks"):
                        diag = json.loads(rec["diagnostics"])
                        for i, (cid, text) in enumerate(zip(json.loads(rec["chunk_ids"]), json.loads(rec["retrieved_contexts"])), 1):
                            ranks = " | ".join(f"{k}: {v.get(cid, '-')}" for k, v in diag.items())
                            st.markdown(f"**{i}. {cid}** — {ranks}")
                            st.caption(text[:400] + ("..." if len(text) > 400 else ""))
                    st.session_state.qa_log.append(rec)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ---------------------------------------------------------------------------
# TAB 2 - batch evaluation: every test question through the selected arms
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload **test_questions.csv** with columns `id, question, reference`. "
                "Every question is run through each selected arm with the same model, prompt and settings.")
    up = st.file_uploader("test_questions.csv", type="csv")
    arms = st.multiselect("Arms to run", ["naive", "advanced"], default=["naive", "advanced"])
    if up is not None:
        qs = pd.read_csv(up).fillna("")
        missing = {"id", "question", "reference"} - set(qs.columns)
        if missing:
            st.error(f"Missing column(s): {sorted(missing)}")
        elif not qs["id"].is_unique:
            st.error("Duplicate ids in the file.")
        else:
            st.write(f"{len(qs)} questions loaded.")
            if st.button("Run evaluation", type="primary"):
                results = {arm: [] for arm in arms}
                bar = st.progress(0.0)
                total, n = len(qs) * len(arms), 0
                for arm in arms:
                    for _, q in qs.iterrows():
                        try:
                            results[arm].append(answer_one(str(q["id"]), q["question"], arm, q["reference"]))
                        except Exception as e:
                            st.error(f"{arm} {q['id']}: {e} - stopped. Download what finished and re-run the rest.")
                            break
                        n += 1
                        bar.progress(n / total, text=f"{arm} · {q['id']}")
                        time.sleep(1)
                st.session_state.batch = results
                st.session_state.qa_log.extend(r for rs in results.values() for r in rs)

    if st.session_state.get("batch"):
        results = st.session_state.batch
        models = {r["model_id"] for rs in results.values() for r in rs}
        if len(models) > 1:
            st.error(f"More than one model answered: {models}. Do not use this run.")
        for arm, rs in results.items():
            ragas = [{"id": r["case_id"], "question": r["query"], "answer": r["answer"],
                      "contexts": json.loads(r["retrieved_contexts"]), "reference": r["ground_truth"],
                      "arm": arm, "model_id": r["model_id"], "temperature": r["temperature"],
                      "chunk_ids": json.loads(r["chunk_ids"]), "timestamp": r["timestamp"]} for r in rs]
            st.download_button(f"⬇️ {arm}_outputs.json ({len(rs)} answers, for RAGAS)",
                               json.dumps(ragas, ensure_ascii=False, indent=1).encode("utf-8"),
                               file_name=f"{arm}_outputs.json", mime="application/json")
        settings = dict(groq_model=GROQ_MODEL, models_returned=sorted(models), temperature=TEMPERATURE,
                        reasoning_effort=REASONING_EFFORT if GROQ_MODEL.startswith("openai/gpt-oss") else None,
                        max_tokens=MAX_TOKENS, embedder=EMBED_MODEL, reranker=RERANKER, dense_top_n=DENSE_TOP_N,
                        bm25_top_n=BM25_TOP_N, rrf_k=RRF_K, rerank_pool=RERANK_POOL_SIZE, top_n_context=TOP_N_CONTEXT,
                        n_chunks=len(chunks), index_built=manifest.get("built"), index_sha256=manifest.get("input_sha256"),
                        run_finished=datetime.now().isoformat(timespec="seconds"))
        st.download_button("⬇️ run_settings.json", json.dumps(settings, indent=1).encode("utf-8"),
                           file_name="run_settings.json", mime="application/json")

# ---------------------------------------------------------------------------
# Session log (lost on refresh - download before closing the tab)
# ---------------------------------------------------------------------------
if st.session_state.qa_log:
    st.divider()
    st.subheader(f"📊 This Session's Q&A Log ({len(st.session_state.qa_log)} records)")
    df_log = pd.DataFrame(st.session_state.qa_log)
    st.dataframe(df_log, width="stretch")
    st.download_button("⬇️ Download Log as CSV (eval_log.csv)", df_log.to_csv(index=False).encode("utf-8"),
                       file_name="eval_log.csv", mime="text/csv")
