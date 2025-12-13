import os
import re
import time
from typing import List, Dict, Tuple

import streamlit as st

from openai import OpenAI

import numpy as np

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

try:
    from pypdf import PdfReader
except Exception as e:
    PdfReader = None


APP_TITLE = "WHO Antibiotic Guide"
APP_SUBTITLE = "AWaRe Clinical Assistant"
DEFAULT_PDF_PATH = "WHOAMR.pdf"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


WHO_SYSTEM_PROMPT = """
You are WHO Antibiotic Guide; AWaRe Clinical Assistant.
Purpose: support rational antibiotic use and antimicrobial stewardship using ONLY the provided WHO AWaRe book context.
Scope: common infections; empiric treatment; when a no antibiotic approach is appropriate; choice of antibiotics; dosage; duration; adults and children.
Safety:
1: Do not diagnose; do not replace clinical judgement; do not replace local or national guidelines.
2: If the answer is not explicitly supported by the provided context; say: "Not found in the WHO AWaRe book context provided."
3: Prefer a no antibiotic approach when the context indicates it.
4: When recommending antibiotics; include dose; route; frequency; and duration if present in context.
5: Keep output concise and clinical; include a short stewardship reminder at the end.
Output format:
A: Answer
B: Key dosing and duration
C: When no antibiotics are appropriate
D: Source excerpts with page numbers
""".strip()


st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")

st.title(f"💬 {APP_TITLE}")
st.caption(APP_SUBTITLE)

st.write(
    "Guideline grounded decision support using the WHO AWaRe antibiotic book; "
    "supports stewardship and rational antibiotic use; does not replace clinical judgement or local protocols."
)

if faiss is None:
    st.error("FAISS is not installed. Install faiss-cpu in your environment.")
if PdfReader is None:
    st.error("pypdf is not installed. Install pypdf in your environment.")

with st.sidebar:
    st.header("Settings")

    st.markdown("API key handling")
    st.caption("Recommended: set OPENAI_API_KEY as an environment variable or Streamlit secret.")

    api_key_from_secrets = None
    try:
        api_key_from_secrets = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key_from_secrets = None

    api_key_from_env = os.environ.get("OPENAI_API_KEY")

    use_manual_key = st.toggle("Enter API key manually", value=False)
    manual_key = None
    if use_manual_key:
        manual_key = st.text_input("OpenAI API Key", type="password")

    openai_api_key = manual_key or api_key_from_secrets or api_key_from_env
    if not openai_api_key:
        st.warning("No API key found. Set OPENAI_API_KEY in secrets or environment; or enable manual entry.")

    chunk_size = st.number_input("Chunk size; characters", min_value=600, max_value=4000, value=1500, step=100)
    chunk_overlap = st.number_input("Chunk overlap; characters", min_value=0, max_value=800, value=200, step=50)
    top_k = st.number_input("Top K retrieved chunks", min_value=2, max_value=10, value=5, step=1)

    st.markdown("Answer style")
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.6, value=0.0, step=0.1)

    st.markdown("Document")
    use_uploaded_pdf = st.toggle("Upload a PDF instead of using local WHOAMR.pdf", value=False)

    uploaded_pdf = None
    if use_uploaded_pdf:
        uploaded_pdf = st.file_uploader("Upload WHO AWaRe PDF", type=["pdf"])


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(pdf_bytes)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _clean_text(text)
        pages.append({"page": i + 1, "text": text})
    return pages


def _extract_pdf_pages_from_path(path: str) -> List[Dict]:
    with open(path, "rb") as f:
        data = f.read()
    return _extract_pdf_pages(data)


def _chunk_text_by_pages(
    pages: List[Dict],
    chunk_size_chars: int,
    overlap_chars: int,
) -> List[Dict]:
    chunks = []
    for p in pages:
        page_num = p["page"]
        text = p["text"]
        if not text:
            continue

        start = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(
                    {
                        "page": page_num,
                        "text": chunk,
                    }
                )
            if end >= n:
                break
            start = max(0, end - overlap_chars)

    return chunks


def _embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """
    Returns: float32 array shape (len(texts), dim)
    """
    vectors = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        vectors.extend(batch_vecs)
    return np.vstack(vectors)


def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def _search_index(
    index: faiss.Index,
    client: OpenAI,
    query: str,
    chunks: List[Dict],
    k: int,
) -> List[Dict]:
    qvec = _embed_texts(client, [query])
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = chunks[int(idx)]
        results.append(
            {
                "score": float(score),
                "page": item["page"],
                "text": item["text"],
            }
        )
    return results


@st.cache_resource(show_spinner=True)
def build_retriever_resources(
    pdf_cache_key: str,
    pdf_bytes: bytes,
    chunk_size_chars: int,
    overlap_chars: int,
    openai_api_key_for_cache: str,
) -> Dict:
    """
    Cached: builds chunks, embeddings, FAISS index.
    pdf_cache_key ensures cache invalidation when PDF changes.
    """
    if not openai_api_key_for_cache:
        raise ValueError("OPENAI_API_KEY is required to build index.")

    client = OpenAI(api_key=openai_api_key_for_cache)

    pages = _extract_pdf_pages(pdf_bytes)
    chunks = _chunk_text_by_pages(pages, chunk_size_chars, overlap_chars)

    texts = [c["text"] for c in chunks]
    vectors = _embed_texts(client, texts)
    index = _build_faiss_index(vectors)

    return {"chunks": chunks, "index": index}


def _make_context_block(hits: List[Dict], max_chars_per_hit: int = 1200) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        excerpt = h["text"]
        if len(excerpt) > max_chars_per_hit:
            excerpt = excerpt[:max_chars_per_hit].rstrip() + " ..."
        blocks.append(f"Source {i}; page {h['page']}:\n{excerpt}")
    return "\n\n".join(blocks)


def _answer_with_citations(
    client: OpenAI,
    query: str,
    hits: List[Dict],
    temperature: float,
) -> str:
    context = _make_context_block(hits)

    user_prompt = f"""
WHO AWaRe book context:
{context}

User question:
{query}

Write the answer following the required output format.
""".strip()

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": WHO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    return stream


def _get_pdf_bytes() -> Tuple[str, bytes]:
    """
    Returns: cache_key, pdf_bytes
    """
    if use_uploaded_pdf and uploaded_pdf is not None:
        data = uploaded_pdf.getvalue()
        cache_key = f"upload:{uploaded_pdf.name}:{len(data)}"
        return cache_key, data

    if os.path.exists(DEFAULT_PDF_PATH):
        with open(DEFAULT_PDF_PATH, "rb") as f:
            data = f.read()
        cache_key = f"local:{DEFAULT_PDF_PATH}:{len(data)}"
        return cache_key, data

    return "missing", b""


def _trim_chat_history(messages: List[Dict], keep_last: int = 8) -> List[Dict]:
    """
    Not used in the RAG call itself; kept for UI history display only.
    """
    if len(messages) <= keep_last * 2:
        return messages
    return messages[-keep_last * 2 :]


if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Status")
    pdf_key, pdf_bytes = _get_pdf_bytes()

    if pdf_key == "missing":
        st.error("WHOAMR.pdf not found and no upload provided. Add WHOAMR.pdf or upload it in the sidebar.")

    if not openai_api_key:
        st.warning("API key missing; add it in secrets or environment; or enable manual entry.")

    if pdf_key != "missing" and openai_api_key:
        try:
            with st.spinner("Indexing or loading cached index"):
                resources = build_retriever_resources(
                    pdf_cache_key=pdf_key,
                    pdf_bytes=pdf_bytes,
                    chunk_size_chars=int(chunk_size),
                    overlap_chars=int(chunk_overlap),
                    openai_api_key_for_cache=openai_api_key,
                )
            st.success("Retriever ready")
            st.caption(f"Chunks indexed: {len(resources['chunks'])}")
        except Exception as e:
            st.error(f"Index build failed: {e}")
            resources = None
    else:
        resources = None

    st.subheader("Disclaimer")
    st.write(
        "Decision support only; based on WHO AWaRe content provided. "
        "Does not replace clinical judgement or local and national prescribing guidelines."
    )

with col1:
    st.subheader("Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about empiric therapy; dosing; duration; or when no antibiotics are appropriate")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if resources is None:
            with st.chat_message("assistant"):
                st.error("Retriever not ready. Check API key and PDF availability in the sidebar.")
        else:
            client = OpenAI(api_key=openai_api_key)

            with st.chat_message("assistant"):
                try:
                    hits = _search_index(
                        index=resources["index"],
                        client=client,
                        query=prompt,
                        chunks=resources["chunks"],
                        k=int(top_k),
                    )

                    if not hits:
                        st.write("Not found in the WHO AWaRe book context provided.")
                    else:
                        stream = _answer_with_citations(
                            client=client,
                            query=prompt,
                            hits=hits,
                            temperature=float(temperature),
                        )

                        response_text = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                        with st.expander("Retrieved sources; page linked"):
                            for i, h in enumerate(hits, start=1):
                                st.markdown(f"Source {i}; page {h['page']}; similarity {h['score']:.3f}")
                                st.write(h["text"][:1500] + (" ..." if len(h["text"]) > 1500 else ""))
                                st.markdown("")

                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.session_state.messages = _trim_chat_history(st.session_state.messages, keep_last=10)
