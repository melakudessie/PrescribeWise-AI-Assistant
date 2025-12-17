import streamlit as st
from openai import OpenAI
import pypdf
import numpy as np
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
APP_TITLE = "PrescribeWise - Health Worker Assistant"
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG Parameters (Hidden from UI)
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 6  # Increased to 6 to ensure enough context for detailed answers

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CSS & STYLING
# ================================
def load_css():
    st.markdown("""
    <style>
        .header-container {
            background: linear-gradient(135deg, #0051A5 0%, #00A3DD 100%);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            color: white;
            text-align: center;
        }
        .header-title {
            font-size: 2em;
            font-weight: 700;
            margin: 0;
            color: white;
        }
        .header-subtitle {
            font-size: 1.1em;
            opacity: 0.9;
            margin-top: 5px;
            color: #E0F2FF;
        }
        .info-box {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #0051A5;
            margin-bottom: 20px;
        }
        .disclaimer-box {
            background-color: #FFF3CD;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #FFC107;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #856404;
        }
        .stButton > button {
            background-color: #0051A5;
            color: white;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

# ================================
# RAG LOGIC (Chunking & Search)
# ================================

def get_pdf_text_by_page(pdf_path):
    """Extracts text from PDF and returns a list of (page_number, text) tuples."""
    path = Path(pdf_path)
    if not path.exists():
        return None, "PDF file not found."
    
    try:
        reader = pypdf.PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
        return pages, None
    except Exception as e:
        return None, str(e)

def create_chunks(pages):
    """
    Creates chunks of ~CHUNK_SIZE characters with CHUNK_OVERLAP.
    Tracks which page(s) the chunk belongs to.
    """
    chunks = []
    
    # Flatten text stream but keep track of page transitions
    full_text_stream = []
    for page_num, text in pages:
        clean_text = text.replace('\n', ' ').strip()
        full_text_stream.append({"text": clean_text, "page": page_num})
        
    # Combine all text while mapping character indices to page numbers
    combined_text = ""
    page_map = [] # List of dictionaries: {"start": int, "end": int, "page": int}
    
    for entry in full_text_stream:
        start_index = len(combined_text)
        combined_text += entry['text'] + " "
        end_index = len(combined_text)
        page_map.append({"start": start_index, "end": end_index, "page": entry['page']})
        
    # Create Chunks
    total_len = len(combined_text)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    
    if total_len == 0:
        return []

    for i in range(0, total_len, step):
        end = min(i + CHUNK_SIZE, total_len)
        chunk_text = combined_text[i:end]
        
        # Find which pages are in this chunk
        chunk_pages = set()
        for mapping in page_map:
            # Check for intersection between chunk range and page range
            if max(i, mapping['start']) < min(end, mapping['end']):
                chunk_pages.add(mapping['page'])
        
        if not chunk_pages:
            continue

        sorted_pages = sorted(list(chunk_pages))
        if len(sorted_pages) == 1:
            page_ref = f"{sorted_pages[0]}"
        else:
            page_ref = f"{sorted_pages[0]}-{sorted_pages[-1]}"
        
        if len(chunk_text) > 100: # Skip tiny chunks
            chunks.append({
                "text": chunk_text,
                "page_ref": page_ref
            })
            
    return chunks

@st.cache_resource
def load_and_index_pdf(pdf_path, _client):
    """Loads PDF, chunks it, and creates embeddings (Cached)."""
    
    # 1. Load Text
    pages, error = get_pdf_text_by_page(pdf_path)
    if error:
        return None, error
        
    # 2. Create Chunks
    raw_chunks = create_chunks(pages)
    
    # 3. Generate Embeddings
    try:
        text_list = [c["text"] for c in raw_chunks]
        embeddings = []
        
        # Batching for API reliability
        batch_size = 100
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            if not batch: 
                continue
            response = _client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            embeddings.extend([data.embedding for data in response.data])
            
        return {
            "chunks": raw_chunks,
            "embeddings": np.array(embeddings)
        }, None
        
    except Exception as e:
        return None, str(e)

def retrieve_context(query, kb, client):
    """Retrieves top K chunks based on cosine similarity."""
    if not kb:
        return []
        
    # Embed query
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_vec = np.array(response.data[0].embedding)
    
    # Calculate Similarity
    sims = np.dot(kb["embeddings"], query_vec)
    
    # Get Top K
    k = min(TOP_K_CHUNKS, len(kb["chunks"]))
    top_indices = np.argsort(sims)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append(kb["chunks"][idx])
        
    return results

# ================================
# UI COMPONENTS
# ================================

def render_header():
    st.markdown("""
    <div class="header-container">
        <div class="header-title">💡 PrescribeWise</div>
        <div class="header-subtitle">Health Worker Assistant • WHO AWaRe Book (2022)</div>
    </div>
    """, unsafe_allow_html=True)

def render_simplified_intro():
    st.markdown("""
    <div class="info-box">
        <h4 style="margin-top:0; color: #0051A5;">👋 Welcome to PrescribeWise</h4>
        <p style="font-size:1.1em;">
            Your AI assistant for appropriate antibiotic prescribing and stewardship. 
            Based exclusively on the <strong>WHO AWaRe Antibiotic Book (2022)</strong>.
        </p>
        <ul style="margin-bottom:0;">
            <li><strong>🟢 Access:</strong> First-line, lower resistance potential.</li>
            <li><strong>🟡 Watch:</strong> Key targets for stewardship actions.</li>
            <li><strong>🔴 Reserve:</strong> Last-resort options for MDR organisms.</li>
        </ul>
    </div>
    
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This tool is for <strong>educational purposes only</strong> for healthcare professionals. 
        It is NOT a substitute for professional clinical judgment. Always refer to official local 
        guidelines and consider individual patient factors.
    </div>
    """, unsafe_allow_html=True)

# ================================
# MAIN APP
# ================================
def main():
    load_css()
    
    # -- Init Session --
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # -- Auth & Setup --
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 OpenAI API Key missing.")
        st.stop()
        
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # -- Load Knowledge Base --
    with st.spinner("Loading WHO Guidelines..."):
        kb, error = load_and_index_pdf("WHOAMR.pdf", client)

    # -- UI Layout --
    render_header()
    
    # Sidebar (Clean - No Settings)
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
        st.markdown("### 🟢 AWaRe Classification")
        st.markdown("""
        **Access:** First choice
        **Watch:** Second choice
        **Reserve:** Last resort
        """)
        st.divider()
        if st.button("🔄 New Consultation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    # Main Content
    if error:
        st.error(f"❌ Error loading knowledge base: {error}")
    elif not kb:
        st.warning("⚠️ 'WHOAMR.pdf' not found. Please upload the document.")
    else:
        # Show toast safely (once per load)
        if "kb_loaded" not in st.session_state:
            st.toast(f"Knowledge Base Loaded: {len(kb['chunks'])} sections indexed.", icon="📚")
            st.session_state.kb_loaded = True

        # Show intro/disclaimer only if chat is empty
        if not st.session_state.messages:
            render_simplified_intro()

        # -- Chat History --
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # -- Chat Input --
        if prompt := st.chat_input("Ex: Treatment for pneumonia in children?"):
            
            # 1. User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # 2. Retrieval
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status = st.status("🔍 Searching guidelines...", expanded=False)
                
                context_chunks = retrieve_context(prompt, kb, client)
                
                # Format context for LLM
                context_str = "\n\n".join(
                    [f"--- [SOURCE: Pages {c['page_ref']}] ---\n{c['text']}" for c in context_chunks]
                )
                
                status.write(f"Found {len(context_chunks)} relevant sections.")
                status.update(label="✅ Search Complete", state="complete")
                
                # 3. Generation (UPDATED FOR DETAIL)
                system_prompt = f"""
You are PrescribeWise, an expert clinical assistant based on the WHO AWaRe Antibiotic Book (2022).

Your goal is to provide comprehensive, actionable, and detailed treatment guidance.

INSTRUCTIONS:
1. **Response Structure:** Organize your answer into clear sections:
   - **Clinical Summary:** Briefly confirm the condition/scenario.
   - **Primary Treatment (Access 🟢):** First-line antibiotic choice with exact dosing and duration.
   - **Alternative Treatment (Watch 🟡):** Second-line options (e.g., for allergy or failure) with dosing/duration.
   - **Comments/Monitoring:** Key clinical pearls, warnings, or monitoring instructions found in the text.
   
2. **Detail Level:** - Provide **specific dosing** for both Adults and Children if available.
   - Include **treatment duration**.
   - Explain the **rationale** if the text provides it.
   - Do NOT be brief. Be thorough.

3. **Citations:** You MUST cite the page numbers provided in the text markers (e.g., [Pages 45-46]) for every major claim.

4. **Classification:** Always label antibiotics clearly with their AWaRe classification:
   - Access (🟢)
   - Watch (🟡)
   - Reserve (🔴)

5. **Limitations:** If the answer is not in the context, state "I cannot find this information in the WHO guidelines."

CONTEXT:
{context_str}
"""
                
                full_response = ""
                try:
                    stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True,
                        temperature=0.3
                    )
                    
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                            
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
