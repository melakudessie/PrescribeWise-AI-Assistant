import os
import re
import io
from typing import List, Dict, Tuple, Optional

import streamlit as st
import numpy as np

from openai import OpenAI

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


APP_TITLE: str = "WHO Antibiotic Guide"
APP_SUBTITLE: str = "AWaRe(Access, Watch, Reserve) Clinical Assistant"
DEFAULT_PDF_PATH: str = "WHOAMR.pdf"

EMBED_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o-mini"

STEWARD_FOOTER: str = (
    "Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; "
    "follow local guidance and clinical judgment."
)

# IMPROVED SYSTEM PROMPT - removed A/B/C/D structure for cleaner output
WHO_SYSTEM_PROMPT: str = """
You are the WHO Antibiotic Guide, an AWaRe (Access, Watch, Reserve) Clinical Assistant.

Purpose: Support rational antibiotic use and antimicrobial stewardship using ONLY the provided WHO AWaRe book context.

Scope: Common infections, empiric treatment at first presentation, when antibiotics are not appropriate, antibiotic choice, dosage, route, frequency, duration for adults and children.

Safety rules:
1. Use ONLY the provided WHO context - do not use outside knowledge
2. If the answer is not explicitly supported by the context, say: "I couldn't find specific information about this in the WHO AWaRe handbook provided."
3. Only recommend avoiding antibiotics if the WHO context explicitly states antibiotics are not needed, not recommended, or should be avoided
4. Do not diagnose, do not replace clinical judgment, do not replace local or national guidelines

Response format:
Write a clear, professional medical reference response with these sections:

**Main Answer:**
- Provide a direct, concise answer to the question (2-3 sentences)
- Include the AWaRe category (Access/Watch/Reserve) when discussing specific antibiotics
- Use clear medical terminology but remain accessible

**Treatment Details:** (only include if the question asks about treatment/dosing)
- Dosing: specific doses with units (e.g., mg/kg, g)
- Route: oral, IV, IM, etc.
- Frequency: every X hours, times daily, etc.
- Duration: number of days
- Format as bullet points for clarity

**When Antibiotics Are NOT Needed:** (only include if WHO context indicates this)
- State clearly if antibiotics are not appropriate
- Provide the WHO justification

**Sources:**
- Cite page numbers from the WHO AWaRe handbook
- Include brief relevant excerpts (1-2 sentences max per source)
- Maximum 3 sources

Guidelines:
- Be concise and direct - avoid unnecessary repetition
- Do not use labels like "A:", "B:", "C:", "D:"
- Use section headers like "**Treatment Details:**" for clarity
- If information is incomplete, acknowledge this clearly
- Focus on practical clinical guidance

Always end your response with:
Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; follow local guidance and clinical judgment.
""".strip()


st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")
st.title(f"💊 {APP_TITLE}")
st.caption(APP_SUBTITLE)

st.write(
    "Guideline grounded decision support using the WHO AWaRe book; "
    "supports antimicrobial stewardship; does not replace clinical judgment or local protocols."
)

if PdfReader is None:
    st.error("Dependency missing: pypdf. Add pypdf to requirements.txt.")
if faiss is None:
    st.error("Dependency missing: faiss. Add faiss-cpu to requirements.txt.")


def ensure_footer(text: str) -> str:
    if not text:
        return STEWARD_FOOTER
    if STEWARD_FOOTER.lower() in text.lower():
        return text
    return (text.rstrip() + "\n\n" + STEWARD_FOOTER).strip()


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    s = re.sub(r"\s*mg\s*/\s*kg\s*/\s*day", " mg/kg/day", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*mg\s*/\s*kg\s*/\s*dose", " mg/kg/dose", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*IV\s*/\s*IM", " IV/IM", s, flags=re.IGNORECASE)

    return s.strip()


def _read_pdf_bytes_from_path(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[Dict]:
    bio = io.BytesIO(pdf_bytes)
    reader = PdfReader(bio)
    pages: List[Dict] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _clean_text(text)
        pages.append({"page": i + 1, "text": text})
    return pages


def _chunk_pages(pages: List[Dict], chunk_size_chars: int, chunk_overlap_chars: int) -> List[Dict]:
    chunks: List[Dict] = []
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
                chunks.append({"page": page_num, "text": chunk})
            if end >= n:
                break
            start = max(0, end - chunk_overlap_chars)

    return chunks


def _embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(vectors).astype(np.float32)


def _build_index(vectors: np.ndarray) -> "faiss.Index":
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def _search(index: "faiss.Index", client: OpenAI, query: str, chunks: List[Dict], k: int) -> List[Dict]:
    qvec = _embed_texts(client, [query])
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, k)

    hits: List[Dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        c = chunks[int(idx)]
        hits.append({"score": float(score), "page": c["page"], "text": c["text"]})
    return hits


def _make_context(hits: List[Dict], max_chars: int = 1500) -> str:
    """Create context from retrieved chunks - increased from 1200 to 1500 chars"""
    blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        excerpt = h["text"]
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + " ..."
        blocks.append(f"[Source {i}, Page {h['page']}]:\n{excerpt}")
    return "\n\n".join(blocks)


def _stream_answer(client: OpenAI, question: str, hits: List[Dict], temperature: float):
    context = _make_context(hits)
    user_prompt = f"""
WHO AWaRe book context:
{context}

User question:
{question}

Provide a clear, well-structured answer following the response format guidelines.
""".strip()

    return client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": WHO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )


def _extract_openai_key(raw: Optional[str]) -> str:
    """
    Extract a valid ASCII key token from secrets or input.
    Prevents UnicodeEncodeError caused by emojis or extra characters in the secret.
    """
    if not raw:
        return ""

    raw = raw.strip().strip('"').strip("'").strip()

    m = re.search(r"(sk-proj-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})", raw)
    if m:
        return m.group(1)

    ascii_only = raw.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    m2 = re.search(r"(sk-proj-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})", ascii_only)
    if m2:
        return m2.group(1)

    return ""


def _get_openai_key_from_secrets_or_env() -> str:
    key = ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("OPENAI_API_KEY", "")

    return _extract_openai_key(key)


def _get_pdf_bytes_from_repo(local_path: str) -> Tuple[str, Optional[bytes], str]:
    if os.path.exists(local_path):
        try:
            data = _read_pdf_bytes_from_path(local_path)
            return f"repo:{local_path}:{len(data)}", data, f"Using PDF from repo: {local_path}"
        except Exception as e:
            return "repo:read_error", None, f"Failed to read repo PDF: {e}"
    return "repo:missing", None, f"Missing PDF in repo root: {local_path}"


@st.cache_resource(show_spinner=True)
def build_retriever(pdf_cache_key: str, pdf_bytes: bytes, chunk_size: int, chunk_overlap: int, openai_api_key: str) -> Dict:
    if PdfReader is None:
        raise RuntimeError("pypdf is not available.")
    if faiss is None:
        raise RuntimeError("faiss is not available.")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY missing or invalid. Use only sk-... in Streamlit Secrets.")

    client = OpenAI(api_key=openai_api_key)

    pages = _read_pdf_pages_from_bytes(pdf_bytes)
    chunks = _chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise RuntimeError("No text extracted from PDF. If the PDF is scanned, use a text based version or add OCR.")

    vectors = _embed_texts(client, [c["text"] for c in chunks])
    index = _build_index(vectors)

    return {"chunks": chunks, "index": index}


with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("**API Key**")
    use_manual = st.toggle("Enter API key manually", value=False)

    manual_raw = ""
    if use_manual:
        manual_raw = st.text_input("OpenAI API Key", type="password")

    openai_api_key = _extract_openai_key(manual_raw) if manual_raw else _get_openai_key_from_secrets_or_env()

    if openai_api_key:
        st.success("✅ API key valid.")
    else:
        st.warning('⚠️ API key not found or invalid. In Streamlit Secrets use: OPENAI_API_KEY = "sk-..."')

    st.divider()

    st.markdown("**📄 Document**")
    st.caption("This app reads WHOAMR.pdf from your GitHub repo; no upload is required.")
    st.caption("Confirm WHOAMR.pdf is in the same folder as streamlit_app.py in GitHub.")

    st.divider()

    st.markdown("**🔍 Retrieval Settings**")
    chunk_size = st.number_input("Chunk size (characters)", min_value=600, max_value=4000, value=1500, step=100)
    chunk_overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=800, value=200, step=50)
    top_k = st.number_input("Top K chunks", min_value=2, max_value=10, value=5, step=1)

    st.divider()

    st.markdown("**🎛️ Model Settings**")
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.6, value=0.0, step=0.1)
    debug = st.toggle("Debug mode (show full errors)", value=False)

    st.divider()

    st.markdown("**💡 Example Questions**")
    example_questions = [
        "What is the first line treatment for pneumonia?",
        "What are reserve antibiotics?",
        "Treatment for UTI in adults?",
        "Amoxicillin dosing for children?",
    ]
    
    for eq in example_questions:
        if st.button(eq, key=f"example_{eq}", use_container_width=True):
            st.session_state["example_question"] = eq

    st.divider()

    st.subheader("⚠️ Disclaimer")
    st.caption(
        "Decision support tool based on WHO AWaRe content provided. "
        "Does not replace clinical judgment or local and national prescribing guidelines."
    )


left, right = st.columns([2, 1])

with right:
    st.subheader("📊 Status")

    pdf_key, pdf_bytes, pdf_status = _get_pdf_bytes_from_repo(DEFAULT_PDF_PATH)
    st.write(pdf_status)

    if not openai_api_key:
        st.error("❌ API key missing or invalid.")
    if pdf_bytes is None:
        st.error("❌ PDF unavailable in repo.")
    if PdfReader is None or faiss is None:
        st.error("❌ Dependencies missing; check requirements.txt.")


resources = None
retriever_error = None

if openai_api_key and pdf_bytes is not None and PdfReader is not None and faiss is not None:
    try:
        with st.spinner("🔨 Preparing retriever; building index on first run; using cache later"):
            resources = build_retriever(
                pdf_cache_key=pdf_key,
                pdf_bytes=pdf_bytes,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                openai_api_key=openai_api_key,
            )
    except Exception as e:
        retriever_error = e
        resources = None


with left:
    st.subheader("💬 Chat")

    if resources is None:
        st.error("❌ Retriever not ready. Check API key and PDF availability in the sidebar.")
        if retriever_error is not None:
            if debug:
                st.exception(retriever_error)
            else:
                st.write(f"Error: {retriever_error}")
        st.stop()

    st.success(f"✅ Retriever ready; chunks indexed: {len(resources['chunks'])}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle example question button clicks
    if "example_question" in st.session_state:
        question = st.session_state["example_question"]
        del st.session_state["example_question"]
        st.rerun()
    else:
        question = st.chat_input("Ask about empiric therapy, dosing, duration, or when no antibiotics are appropriate...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        client = OpenAI(api_key=openai_api_key)

        with st.chat_message("assistant"):
            try:
                hits = _search(
                    index=resources["index"],
                    client=client,
                    query=question,
                    chunks=resources["chunks"],
                    k=int(top_k),
                )

                if not hits:
                    msg = ensure_footer(
                        "I couldn't find specific information about this in the WHO AWaRe handbook provided.\n\n"
                        "**Try:**\n"
                        "- Rephrasing with more general terms\n"
                        "- Asking about the condition rather than a specific drug\n"
                        "- Using example questions from the sidebar"
                    )
                    st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    stream = _stream_answer(client, question, hits, float(temperature))
                    answer_text = st.write_stream(stream)
                    answer_text = ensure_footer(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})

                    # Improved sources display
                    with st.expander("📚 View Retrieved Sources", expanded=False):
                        st.caption(f"Retrieved {len(hits)} most relevant chunks from the WHO AWaRe handbook:")
                        for i, h in enumerate(hits, start=1):
                            st.markdown(f"**Source {i}** | Page {h['page']} | Similarity: {h['score']:.3f}")
                            excerpt = h["text"][:800]
                            st.text(excerpt + (" ..." if len(h["text"]) > 800 else ""))
                            if i < len(hits):
                                st.divider()

            except Exception as e:
                if debug:
                    st.exception(e)
                else:
                    st.error(f"❌ Request failed: {e}")
                    st.info("Enable 'Debug mode' in the sidebar to see full error details.")

    # Limit chat history to prevent memory issues
    if len(st.session_state.messages) > 24:
        st.session_state.messages = st.session_state.messages[-24:]

# Footer
st.divider()
st.caption("WHO AWaRe Antibiotic Guide | Decision support tool | Always follow local protocols and clinical judgment")
