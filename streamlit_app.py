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
PDF_FILE_PATH = "WHOAMR.pdf"  # The file must exist in the repo

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="PrescribeWise Assistant", page_icon="🩺", layout="wide")
st.title("🩺 PrescribeWise: Health Worker Assistant")

# --- 2. CREDENTIALS & FILE CHECK ---
# Check API Key in Secrets
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 OpenAI API Key not found in Secrets. Please add it to `.streamlit/secrets.toml`.")
    st.stop()

# Check if PDF exists in the repo
if not os.path.exists(PDF_FILE_PATH):
    st.error(f"🚨 File not found: `{PDF_FILE_PATH}`. Please ensure the PDF is committed to your GitHub repository in the same folder as this script.")
    st.stop()

# --- 3. CACHED PDF PROCESSING ---
# We no longer need temp files because the file is permanent in the repo
@st.cache_resource(show_spinner=False)
def load_knowledge_base(key):
    try:
        # Load directly from the local repo path
        loader = PyPDFLoader(PDF_FILE_PATH)
        docs = loader.load()
        
        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # Embed
        embeddings = OpenAIEmbeddings(openai_api_key=key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        raise e

# --- 4. MAIN APP LOGIC ---

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Sidebar (Simplified)
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This assistant uses the WHO Antimicrobial Resistance guidelines.")
    st.success("✅ Knowledge Base Loaded")
    st.success("✅ API Key Connected")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Process Knowledge Base (Cached)
# This runs once on startup and stays cached
with st.spinner("Initializing knowledge base..."):
    try:
        vectorstore = load_knowledge_base(api_key)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        st.stop()

# --- 5. BUILD THE MODERN CHAIN (LCEL) ---
template = """You are a helpful medical assistant called PrescribeWise.
Answer the question based ONLY on the following context from the WHO guidelines.
If the answer is not in the context, say "I cannot find this information in the guidelines."
Always cite the page number if available in the context.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

if user_input := st.chat_input("Ask about antibiotic dosages, treatments, etc..."):
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting guidelines..."):
            try:
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
