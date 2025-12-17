import streamlit as st
from openai import OpenAI, AuthenticationError
import base64
from pathlib import Path
import pypdf
import numpy as np

# ================================
# CONFIGURATION
# ================================
APP_TITLE = "PrescribeWise - Health Worker Assistant"
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
# Number of pages to retrieve per query (adjust based on density of info)
TOP_K_PAGES = 5 

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💡",
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
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
            color: white;
            text-align: center;
        }
        .header-title {
            font-size: 2.2em;
            font-weight: 700;
            margin: 0;
        }
        .header-subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .stButton > button {
            background-color: #0051A5;
            color: white;
        }
        .citation-box {
            font-size: 0.85em;
            color: #666;
            border-left: 3px solid #ddd;
            padding-left: 10px;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# ================================
# KNOWLEDGE BASE LOGIC (RAG)
# ================================

@st.cache_resource
def process_document(pdf_path, _client):
    """
    1. Loads PDF.
    2. Splits by page.
    3. Generates embeddings for each page.
    4. Returns a dictionary with text chunks and their vector embeddings.
    """
    path = Path(pdf_path)
    if not path.exists():
        return None, "PDF file not found."

    status_container = st.empty()
    status_container.info("🔄 Indexing WHO AWaRe Book (700+ pages)... This happens only once.")

    try:
        # 1. Read PDF
        reader = pypdf.PdfReader(path)
        chunks = []
        
        # Extract text per page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:  # Skip empty pages
                # We prepend page number to the text so the model knows where it is
                chunks.append({
                    "id": i,
                    "page_num": i + 1,
                    "text": f"PAGE {i+1}:\n{text}"
                })

        # 2. Generate Embeddings (Batching to be safe)
        text_list = [c["text"] for c in chunks]
        embeddings = []
        
        # OpenAI usually allows large batches, but safe batching is 100
        batch_size = 100
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            response = _client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        # 3. Store in structure
        knowledge_base = {
            "chunks": chunks,
            "embeddings": np.array(embeddings) # Convert to numpy for fast math
        }
        
        status_container.empty()
        return knowledge_base, None

    except Exception as e:
        status_container.error(f"Error processing document: {e}")
        return None, str(e)

def search_knowledge_base(query, knowledge_base, client, top_k=5):
    """
    Semantic search: embeds the query and finds the most similar pages.
    """
    if not knowledge_base:
        return []

    # 1. Embed Query
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_embedding = np.array(response.data[0].embedding)

    # 2. Calculate Cosine Similarity
    # (Dot product of normalized vectors)
    # We assume OpenAI embeddings are normalized, so dot product is sufficient
    chunk_embeddings = knowledge_base["embeddings"]
    similarities = np.dot(chunk_embeddings, query_embedding)

    # 3. Get Top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # 4. Retrieve results
    results = []
    for idx in top_indices:
        results.append(knowledge_base["chunks"][idx])
        
    return results

# ================================
# UI COMPONENTS
# ================================
def render_header():
    st.markdown("""
    <div class="header-container">
        <div style="font-size: 3em; margin-bottom: 10px;">🩺</div>
        <div class="header-title">PrescribeWise</div>
        <div class="header-subtitle">WHO AWaRe Antibiotic Assistant</div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.header("About")
        st.info(
            "This AI uses **Retrieval-Augmented Generation (RAG)** to search "
            "the 700-page WHO AWaRe book and provide precise answers with citations."
        )
        st.divider()
        st.markdown("### 🚦 AWaRe Categories")
        st.markdown("🟢 **Access:** First choice, low resistance.")
        st.markdown("🟡 **Watch:** Higher resistance potential.")
        st.markdown("🔴 **Reserve:** Last resort.")
        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

# ================================
# MAIN APPLICATION
# ================================
def main():
    load_css()
    
    # --- 1. SETUP ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a polite greeting from the assistant
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I am ready to help with antibiotic guidelines from the WHO AWaRe book. Ask me about treatments, dosing, or classifications."
        })

    # Validate API Key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 OpenAI API Key missing in `.streamlit/secrets.toml`.")
        st.stop()
    
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        st.stop()

    # --- 2. LOAD KNOWLEDGE BASE (Cached) ---
    kb, error = process_document("WHOAMR.pdf", client)
    
    render_header()
    render_sidebar()

    if error:
        st.error(f"❌ Could not load Knowledge Base: {error}")
        st.warning("Ensure 'WHOAMR.pdf' is in the same directory.")
    else:
        if kb:
            st.toast("✅ Knowledge Base Active (700+ Pages)", icon="📚")

    # --- 3. CHAT INTERFACE ---
    
    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle Input
    if prompt := st.chat_input("Ex: First-line treatment for pneumonia in children?"):
        
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Logic
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # A. SEARCH (Retrieval)
            status_text = st.status("🔍 Searching guidelines...", expanded=False)
            try:
                if kb:
                    # Find relevant pages
                    relevant_chunks = search_knowledge_base(prompt, kb, client, top_k=TOP_K_PAGES)
                    
                    # Construct Context String
                    context_text = "\n\n".join([f"--- [SOURCE: Page {c['page_num']}] ---\n{c['text']}" for c in relevant_chunks])
                    status_text.write(f"Found {len(relevant_chunks)} relevant pages.")
                else:
                    context_text = "No document loaded."
                    
                status_text.update(label="✅ Search Complete", state="complete", expanded=False)
                
                # B. GENERATE (Completion)
                system_prompt = f"""
You are an expert medical assistant based on the WHO AWaRe Antibiotic Book (2022).

INSTRUCTIONS:
1. Use the provided "CONTEXT" to answer the user's question.
2. If the answer is found in the context, you MUST cite the page number, e.g., "(Page 150)".
3. Categorize antibiotics Mentioned (Access/Watch/Reserve) if relevant.
4. If the answer is NOT in the context, admit it politely. Do not make up medical advice.
5. Be concise and structured (use bolding for drug names and doses).

CONTEXT:
{context_text}
"""
                
                stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    temperature=0.2 # Low temperature for factual accuracy
                )
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                status_text.update(label="❌ Error", state="error")
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
