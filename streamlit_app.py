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
APP_TITLE = "PrescribeWise - Health Worker Assistant"

# --- 1. PAGE & BRANDING CONFIGURATION ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main Header Styling */
    .header-container {
        background: linear-gradient(90deg, #005c97 0%, #363795 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 5px;
    }
    
    /* Disclaimer Box Styling */
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        color: #856404;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. UI LAYOUT ---

# Custom Header
st.markdown(f"""
    <div class="header-container">
        <div class="header-title">🩺 PrescribeWise</div>
        <div class="header-subtitle">AI-Powered Assistant for WHO Antimicrobial Guidelines</div>
    </div>
""", unsafe_allow_html=True)

# Professional Disclaimer
st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong><br>
        This AI tool assists healthcare professionals by retrieving information solely from the 
        <em>WHO AWaRe Antibiotic Book</em>. It does <strong>not</strong> replace professional medical judgment. 
        Always verify dosages and treatment protocols with local guidelines.
    </div>
""", unsafe_allow_html=True)

# --- 4. CREDENTIALS & FILE CHECK ---
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 API Key missing! Please add `OPENAI_API_KEY` to your Streamlit Secrets.")
    st.stop()

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"🚨 Guidelines file not found: `{PDF_FILE_PATH}`. Please upload it to your GitHub repository.")
    st.stop()

# --- 5. CACHED KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(key):
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        docs = loader.load()
        
        # Splitter optimized for medical guidelines
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "●", "•", "-", " "]
        )
        splits = splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings(openai_api_key=key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        raise e

# --- 6. MAIN LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=50)
    st.header("App Settings")
    st.divider()
    
    st.markdown("### 🚦 AWaRe Color Legend")
    st.markdown(":green[**🟢 First Choice (Access)**]")
    st.markdown(":orange[**🟡 Second Choice (Watch)**]")
    st.markdown(":red[**🔴 Last Resort (Reserve)**]")
    
    st.divider()
    
    st.markdown("### 📚 Source Material")
    st.info("WHO AWaRe (Access, Watch, Reserve) Antibiotic Book (2022)")
    
    st.divider()
    
    if st.button("🔄 Start New Consultation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Load Database
with st.spinner("Initializing medical knowledge base..."):
    try:
        vectorstore = load_knowledge_base(api_key)
        # RETRIEVER: K=6 for detailed context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        st.stop()

# --- 7. COLOR-CODED PROMPT ENGINEERING ---
template = """You are PrescribeWise, an expert medical assistant based on the WHO AWaRe Antibiotic Book.

INSTRUCTIONS:
1. Answer the question comprehensively using ONLY the context provided below.
2. **COLOR CODING RULES:** You must use Streamlit Markdown colors for treatment lines:
   - For **First Choice / Access** antibiotics, format the line like this: :green[**🟢 First Choice:** Drug Name, Dosage, Duration]
   - For **Second Choice / Watch** antibiotics, format the line like this: :orange[**🟡 Second Choice:** Drug Name, Dosage, Duration]
   - For **Reserve / Last Resort** antibiotics, format the line like this: :red[**🔴 Reserve:** Drug Name, Dosage, Duration]
   - For **Comments/Warnings**, use standard text.

3. Include dosages (Adult & Pediatric) and duration if available.
4. If the answer is not in the context, state: "I cannot find this specific information in the WHO guidelines."
5. **CITATION:** Cite the page number for every section (e.g., [Page 45]).

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

# --- 8. CHAT INTERFACE ---
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ex: What is the treatment for acute otitis media in children?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Consulting WHO guidelines..."):
            try:
                # Retrieve and Format
                relevant_docs = retriever.invoke(user_input)
                formatted_context = format_docs(relevant_docs)
                
                # Stream Response
                response_container = st.empty()
                full_response = ""
                
                # Chain with manually injected context (to allow streaming + source retrieval)
                stream_chain = (
                    {"context": lambda x: formatted_context, "question": RunnablePassthrough()}
                    | prompt 
                    | llm 
                    | StrOutputParser()
                )
                
                for chunk in stream_chain.stream(user_input):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                
                # View Evidence Expander
                with st.expander("🔍 View Clinical Evidence (Source Text)"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', '?')})**")
                        st.caption(doc.page_content)
                        st.divider()
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
