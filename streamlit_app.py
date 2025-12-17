import streamlit as st
import tempfile
import os

# --- MODERN IMPORTS (STABLE) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="PrescribeWise Assistant", page_icon="🩺", layout="wide")
st.title("🩺 PrescribeWise: Health Worker Assistant")

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload Medical Guidelines (PDF)", type="pdf")

# --- 3. CACHED PDF PROCESSING ---
@st.cache_resource(show_spinner=False)
def process_pdf(file, key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        # Load
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # Embed
        embeddings = OpenAIEmbeddings(openai_api_key=key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 4. MAIN APP LOGIC ---
if not api_key:
    st.info("👋 Please enter your OpenAI API Key to continue.")
    st.stop()

if not uploaded_file:
    st.info("📄 Please upload the WHOAMR.pdf file.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF (Cached)
with st.spinner("Processing guidelines..."):
    try:
        vectorstore = process_pdf(uploaded_file, api_key)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.stop()

# --- 5. BUILD THE MODERN CHAIN (LCEL) ---
# This replaces RetrievalQA and avoids the import errors
template = """You are a helpful medical assistant called PrescribeWise.
Answer the question based ONLY on the following context.
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

# The LCEL Chain (Source -> Format -> Prompt -> LLM -> String)
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

if user_input := st.chat_input("Ask about antibiotic dosages..."):
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Stream the response for better UX
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
