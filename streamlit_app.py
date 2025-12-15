import os
import re
import io
from typing import List, Dict, Tuple, Optional
from datetime import datetime, date, timedelta

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

# Rate limiting settings
DAILY_QUERY_LIMIT: int = 100  # Adjust based on your budget
QUERIES_PER_SESSION: int = 20  # Prevent abuse in single session

STEWARD_FOOTER: str = (
    "Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; "
    "follow local guidance and clinical judgment."
)

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

# Banner for free access
st.info("🎁 This app is **free to use** - no API key required! Funded for educational purposes.", icon="ℹ️")

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


def init_rate_limit_state():
    """Initialize rate limiting state"""
    if 'rate_limit' not in st.session_state:
        st.session_state.rate_limit = {
            'daily_count': 0,
            'session_count': 0,
            'last_reset': date.today(),
        }
    
    # Reset daily counter if it's a new day
    if st.session_state.rate_limit['last_reset'] < date.today():
        st.session_state.rate_limit['daily_count'] = 0
        st.session_state.rate_limit['last_reset'] = date.today()


def check_rate_limit() -> bool:
    """
    Check if user has exceeded rate limits
    Returns True if OK to proceed, False if limit exceeded
    """
    init_rate_limit_state()
    
    rl = st.session_state.rate_limit
    
    # Check daily limit
    if rl['daily_count'] >= DAILY_QUERY_LIMIT:
        st.error(f"⚠️ Daily limit of {DAILY_QUERY_LIMIT} queries reached.")
        st.info(
            "The daily limit helps us keep this service free for everyone. "
            "Limit resets at midnight UTC. Please try again tomorrow!"
        )
        return False
    
    # Check session limit (prevent abuse)
    if rl['session_count'] >= QUERIES_PER_SESSION:
        st.warning(
            f"⚠️ You've reached the session limit of {QUERIES_PER_SESSION} queries. "
            "Please refresh the page to continue."
        )
        return False
    
    return True


def increment_rate_limit():
    """Increment rate limit counters"""
    init_rate_limit_state()
    st.session_state.rate_limit['daily_count'] += 1
    st.session_state.rate_limit['session_count'] += 1


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
    """Extract a valid ASCII key token from secrets or input"""
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
    """Get API key from Streamlit secrets or environment"""
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
            return f"repo:{local_path}:{len(data)}", data, f"✅ Using PDF from repo: {local_path}"
        except Exception as e:
            return "repo:read_error", None, f"❌ Failed to read repo PDF: {e}"
    return "repo:missing", None, f"❌ Missing PDF in repo root: {local_path}"


@st.cache_resource(show_spinner=True)
def build_retriever(pdf_cache_key: str, pdf_bytes: bytes, chunk_size: int, chunk_overlap: int, openai_api_key: str) -> Dict:
    if PdfReader is None:
        raise RuntimeError("pypdf is not available.")
    if faiss is None:
        raise RuntimeError("faiss is not available.")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY missing or invalid.")

    client = OpenAI(api_key=openai_api_key)

    pages = _read_pdf_pages_from_bytes(pdf_bytes)
    chunks = _chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise RuntimeError("No text extracted from PDF.")

    vectors = _embed_texts(client, [c["text"] for c in chunks])
    index = _build_index(vectors)

    return {"chunks": chunks, "index": index}


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show usage stats
    init_rate_limit_state()
    rl = st.session_state.rate_limit
    
    daily_remaining = DAILY_QUERY_LIMIT - rl['daily_count']
    session_remaining = QUERIES_PER_SESSION - rl['session_count']
    
    st.metric("Queries Remaining Today", f"{daily_remaining}/{DAILY_QUERY_LIMIT}")
    st.metric("Queries This Session", f"{rl['session_count']}/{QUERIES_PER_SESSION}")
    
    st.caption(f"Resets: {date.today() + timedelta(days=1)} 00:00 UTC")

    st.divider()

    st.markdown("**📄 Document**")
    st.caption("Using WHO AWaRe handbook (WHOAMR.pdf)")

    st.divider()

    st.markdown("**🔍 Advanced Settings**")
    with st.expander("Retrieval & Model Settings"):
        chunk_size = st.number_input("Chunk size", min_value=600, max_value=4000, value=1500, step=100)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=800, value=200, step=50)
        top_k = st.number_input("Top K chunks", min_value=2, max_value=10, value=5, step=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=0.6, value=0.0, step=0.1)

    debug = st.toggle("Debug mode", value=False)

    st.divider()

    st.markdown("**💡 Example Questions**")
    example_questions = [
        "First line treatment for pneumonia?",
        "What are reserve antibiotics?",
        "UTI treatment in adults?",
        "Amoxicillin dosing for children?",
    ]
    
    for eq in example_questions:
        if st.button(eq, key=f"ex_{eq}", use_container_width=True):
            st.session_state["example_q"] = eq

    st.divider()

    st.markdown("**ℹ️ About This Tool**")
    st.caption(
        "Free decision support based on WHO AWaRe handbook. "
        "Does not replace clinical judgment or local guidelines."
    )
    st.caption("Funded for educational use.")

# Get API key from secrets
openai_api_key = _get_openai_key_from_secrets_or_env()

# Main content
left, right = st.columns([2, 1])

with right:
    st.subheader("📊 Status")

    pdf_key, pdf_bytes, pdf_status = _get_pdf_bytes_from_repo(DEFAULT_PDF_PATH)
    st.write(pdf_status)

    if not openai_api_key:
        st.error("❌ API key not configured in app secrets.")
    if pdf_bytes is None:
        st.error("❌ PDF unavailable.")
    if PdfReader is None or faiss is None:
        st.error("❌ Dependencies missing.")

resources = None
retriever_error = None

if openai_api_key and pdf_bytes is not None and PdfReader is not None and faiss is not None:
    try:
        with st.spinner("🔨 Building index..."):
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
        st.error("❌ Retriever not ready.")
        if retriever_error is not None:
            if debug:
                st.exception(retriever_error)
            else:
                st.write(f"Error: {retriever_error}")
        st.stop()

    st.success(f"✅ Ready | {len(resources['chunks'])} chunks indexed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle example questions
    if "example_q" in st.session_state:
        question = st.session_state["example_q"]
        del st.session_state["example_q"]
    else:
        question = st.chat_input("Ask about antibiotic treatment...")

    if question:
        # Check rate limit BEFORE processing
        if not check_rate_limit():
            st.stop()
        
        # Increment counter
        increment_rate_limit()
        
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
                        "I couldn't find specific information about this in the WHO AWaRe handbook.\n\n"
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

                    with st.expander("📚 View Sources", expanded=False):
                        for i, h in enumerate(hits, start=1):
                            st.markdown(f"**Source {i}** | Page {h['page']} | Score: {h['score']:.3f}")
                            st.text(h["text"][:800] + ("..." if len(h["text"]) > 800 else ""))
                            if i < len(hits):
                                st.divider()

            except Exception as e:
                # Decrement counter on error (don't penalize for errors)
                st.session_state.rate_limit['session_count'] -= 1
                st.session_state.rate_limit['daily_count'] -= 1
                
                if debug:
                    st.exception(e)
                else:
                    st.error(f"❌ Request failed: {e}")

    # Limit chat history
    if len(st.session_state.messages) > 24:
        st.session_state.messages = st.session_state.messages[-24:]

st.divider()
st.caption("WHO AWaRe Antibiotic Guide | Free for educational use | Always follow local protocols")
