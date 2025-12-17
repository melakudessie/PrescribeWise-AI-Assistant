import streamlit as st
import os

# --- MODERN IMPORTS (STABLE) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONSTANTS ---
PDF_FILE_PATH = "WHOAMR.pdf"  # File must be in the repo root

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="PrescribeWise Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0E1117;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4F4F4F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #FFF4E5;
        border-left: 5px solid #FFA500;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="main-header">🩺 PrescribeWise</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Assistant for WHO Antimicrobial Guidelines</div>', unsafe_allow_html=True)

# --- DISCLAIMER ---
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
    This AI tool is designed to assist healthcare professionals by retrieving information from the 
    <em>WHO AWaRe (Access, Watch, Reserve) Antibiotic Book</em>. 
    It is <strong>NOT</strong> a substitute for professional medical judgment. 
    Always verify dosages and treatment plans against the primary guidelines and local protocols.
</div>
""", unsafe_allow_html=True)

# --- 2. CREDENTIALS & FILE CHECK ---
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 OpenAI API Key missing! Please add it to Streamlit Secrets.")
    st.stop()

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"🚨 File not found: `{PDF_FILE_PATH}`. Please ensure it is in your GitHub repo.")
    st.stop()

# --- 3. CACHED KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(key):
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        docs = loader.load()
        
        # Optimized splitting for medical context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", "●", "•", ".", " "]
        )
        splits = splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings(openai_api_key=key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        raise e

# --- 4. MAIN LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("ℹ️ About App")
    st.info("""
    **Source Material:** WHO AWaRe Antibiotic Book (2022)
    
    **Capabilities:** - Treatment guidelines
    - Pediatric & Adult dosing
    - AWaRe Classification
    """)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.caption(f"App Version 2.1 | Model: GPT-4")

# Load DB
with st.spinner("Initializing medical knowledge base..."):
    try:
        vectorstore = load_knowledge_base(api_key)
        # RETRIEVER CONFIG: k=6 for more detailed context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        st.stop()

# --- 5. DETAILED PROMPT CHAIN ---
template = """You are PrescribeWise, an expert medical assistant based on the WHO AWaRe Antibiotic Book.

INSTRUCTIONS:
1. Answer the question comprehensively using ONLY the context provided below.
2. If the query asks for treatment, include:
   - First Choice Antibiotics
   - Second Choice / Alternatives
   - Dosages (Adult & Pediatric if available)
   - Duration of therapy
3. Use bullet points for readability.
4. If the answer is not in the context, state: "I cannot find this specific information in the provided WHO guidelines."
5. **CITATION:** You MUST cite the page number for every major claim (e.g., [Page 45]).

CONTEXT:
{context}

QUESTION:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)

def format_docs(docs):
    return "\n\n".join(f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ex: What is the treatment for severe pneumonia in children?"):
    # User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant Message
    with st.chat_message("assistant"):
        with st.spinner("Consulting WHO guidelines..."):
            try:
                response_container = st.empty()
                full_response = ""
                
                # Stream response
                for chunk in rag_chain.stream(user_input):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
