"""
WHO AWaRe Antibiotics Support Chatbot
Based ONLY on WHO AWaRe (Access, Watch, Reserve) Classification Document
Version: 3.2 - Final with Proper Citations
"""

import streamlit as st
from openai import OpenAI
import base64
from pathlib import Path
import PyPDF2

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="WHO AWaRe Antibiotics Bot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# PDF LOADER
# ================================
@st.cache_data
def load_pdf_text(pdf_path="WHOAMR.pdf"):
    """Load and extract text from WHO AWaRe PDF"""
    try:
        if not Path(pdf_path).exists():
            return None, f"PDF not found at {pdf_path}"
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return text, None
    except Exception as e:
        return None, str(e)

# ================================
# LOGO HELPER
# ================================
def get_logo_base64():
    """Convert logo to base64"""
    logo_path = Path("logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .header-container {
        background: linear-gradient(135deg, #0051A5 0%, #00A3DD 100%);
        padding: 40px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .logo-img {
        width: 120px;
        height: 120px;
        margin: 0 auto 15px;
        display: block;
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    .header-title {
        color: white;
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #E0F2FF;
        font-size: 1.2em;
        margin-top: 10px;
    }
    
    .description-box {
        background-color: #F8F9FA;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #0051A5;
        margin: 20px 0;
    }
    
    .disclaimer-box {
        background-color: #FFF3CD;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #FFC107;
        margin: 20px 0;
    }
    
    .source-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #1976D2;
        margin: 20px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0051A5 0%, #00A3DD 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
def create_header():
    logo_base64 = get_logo_base64()
    
    if logo_base64:
        st.markdown(f"""
        <div class="header-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="WHO Logo">
            <h1 class="header-title">WHO AWaRe Antibiotics Chatbot</h1>
            <p class="header-subtitle">Based on WHO AWaRe Classification Document</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="header-container">
            <div style="font-size: 3.5em; margin-bottom: 10px;">💊 📚</div>
            <h1 class="header-title">WHO AWaRe Antibiotics Chatbot</h1>
            <p class="header-subtitle">Based on WHO AWaRe Classification Document</p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# DESCRIPTION
# ================================
def show_description():
    st.markdown("""
    <div class="description-box">
        <h3 style="color: #0051A5;">📖 About This Chatbot</h3>
        <p style="font-size: 1.1em; line-height: 1.8;">
            This chatbot provides detailed information based <strong>exclusively</strong> on the 
            <strong>WHO AWaRe (Access, Watch, Reserve) Classification Document</strong>.
        </p>
        <h4 style="color: #0051A5; margin-top: 20px;">What You'll Get:</h4>
        <ul style="font-size: 1.05em; line-height: 2;">
            <li>🟢 <strong>ACCESS Group</strong> - First-line, first-choice antibiotics</li>
            <li>🟡 <strong>WATCH Group</strong> - Second-line alternatives</li>
            <li>🔴 <strong>RESERVE Group</strong> - Last-resort antibiotics</li>
            <li>📚 <strong>Citations with Page Numbers</strong> - All responses cite the guideline with specific pages</li>
            <li>🎯 <strong>Focused Answers</strong> - Get exactly what you ask for</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ================================
# SOURCE DOCUMENT INFO
# ================================
def show_source_info(pdf_loaded):
    status_emoji = "✅" if pdf_loaded else "⚠️"
    status_text = "Loaded Successfully" if pdf_loaded else "Using AI Knowledge Base"
    status_color = "#28A745" if pdf_loaded else "#FFC107"
    
    st.markdown(f"""
    <div class="source-box">
        <h3 style="color: #1976D2;">📚 Source Document</h3>
        <p style="font-size: 1.1em; line-height: 1.8;">
            <strong>Document:</strong> WHO AWaRe (Access, Watch, Reserve) Classification of Antibiotics<br>
            <strong>File:</strong> WHOAMR.pdf<br>
            <strong>Status:</strong> <span style="color: {status_color};">{status_emoji} {status_text}</span>
        </p>
        <p style="font-size: 0.95em; color: #555; margin-top: 15px;">
            All responses include citations with specific page numbers from the guideline.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# DISCLAIMER
# ================================
def show_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        <h3 style="color: #856404;">⚠️ Important Medical Disclaimer</h3>
        <p style="font-size: 1.1em; line-height: 1.6; color: #856404;">
            This chatbot is for <strong>EDUCATIONAL purposes ONLY</strong>
        </p>
        <ul style="font-size: 1em; line-height: 1.8; color: #856404;">
            <li>❌ NOT a substitute for professional medical advice</li>
            <li>✅ Based on WHO AWaRe classification guideline</li>
            <li>✅ Always consult healthcare professionals</li>
            <li>✅ Consider local resistance patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
def create_sidebar():
    with st.sidebar:
        logo_base64 = get_logo_base64()
        if logo_base64:
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{logo_base64}" style="width: 100px; border-radius: 50%; border: 3px solid #0051A5;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; font-size: 2.5em; margin: 20px 0;">💊</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 WHO AWaRe Groups")
        
        with st.expander("🟢 ACCESS Group", expanded=True):
            st.markdown("""
            **First-line, first-choice antibiotics**
            
            - Narrow spectrum
            - Lower resistance risk
            - Should be widely available
            """)
        
        with st.expander("🟡 WATCH Group"):
            st.markdown("""
            **Second-line alternatives**
            
            - Broader spectrum
            - Higher resistance potential
            - Prioritized stewardship targets
            """)
        
        with st.expander("🔴 RESERVE Group"):
            st.markdown("""
            **Last-resort antibiotics**
            
            - Reserved for specific cases
            - Highest resistance concern
            """)
        
        st.divider()
        
        st.markdown("### 📊 Session Stats")
        if "messages" in st.session_state:
            msg_count = len([m for m in st.session_state.messages if m["role"] != "system"])
            st.metric("Messages", msg_count)
        else:
            st.metric("Messages", 0)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.display_messages = []
            st.session_state.show_info = True
            st.rerun()

# ================================
# LOAD PDF AND INITIALIZE
# ================================

# Load PDF content
pdf_text, pdf_error = load_pdf_text("WHOAMR.pdf")
pdf_loaded = pdf_text is not None

# Show PDF loading status in sidebar
if pdf_loaded:
    st.sidebar.success("✅ WHOAMR.pdf loaded")
else:
    st.sidebar.warning("⚠️ WHOAMR.pdf not found")

# ================================
# API KEY CHECK
# ================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ OpenAI API key not found!")
    st.stop()

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"❌ Error initializing OpenAI: {str(e)}")
    st.stop()

# ================================
# SYSTEM PROMPT WITH PROPER CITATIONS
# ================================
if "messages" not in st.session_state:
    # Truncate PDF text for context
    pdf_context = ""
    if pdf_loaded:
        pdf_context = f"\n\nWHO AWaRe DOCUMENT CONTENT:\n{pdf_text[:10000]}\n[... document continues ...]"
    
    st.session_state.messages = [
        {
            "role": "system",
            "content": f"""You are an expert assistant specialized in the WHO AWaRe (Access, Watch, Reserve) Classification of Antibiotics.

CRITICAL: Base ALL responses EXCLUSIVELY on the WHO AWaRe Classification Document.

{pdf_context}

RESPONSE RULES:

1. **ANSWER ONLY WHAT IS ASKED:**
   - If user asks ONLY about first-line → provide ONLY first-line information
   - If user asks ONLY about second-line → provide ONLY second-line information
   - If user asks about alternatives → then provide alternatives
   - DO NOT automatically include all options unless asked

2. **BE DETAILED AND COMPREHENSIVE:**
   - Provide thorough information for what is asked
   - Include specific dosages when available
   - Include duration of treatment
   - Include age-specific information
   - Include contraindications and safety
   - Include monitoring requirements
   - Make responses substantive (400-800 words when appropriate)

3. **WHO AWaRe CLASSIFICATION:**
   
   🟢 **ACCESS** = First-line, first-choice
   🟡 **WATCH** = Second-line alternatives  
   🔴 **RESERVE** = Last-resort

4. **CITATION FORMAT - CRITICAL:**

   EVERY response MUST end with a proper citation referencing the WHO guideline document with page numbers.
   
   Format:
   ```
   ---
   **Reference:** WHO AWaRe (Access, Watch, Reserve) Classification of Antibiotics for Evaluation and Monitoring of Use, 2021
   **Pages:** [specific page numbers or sections where the information can be found]
   ```
   
   Examples:
   - **Pages:** 15-17 (if specific pages known)
   - **Pages:** Section 3.2, Respiratory Infections
   - **Pages:** Table 1, ACCESS group antibiotics
   - **Pages:** Annex 2, Pediatric dosing
   
   DO NOT include:
   - GitHub repository links
   - URLs
   - Web addresses
   
   ONLY cite the guideline document itself with page/section references.

5. **STRUCTURE RESPONSES:**

   For first-line questions:
   ```
   ## First-Line Treatment for [Condition]
   
   ### 🟢 ACCESS Group Recommendation
   
   **[Antibiotic Name]** (WHO AWaRe: ACCESS)
   
   **Dosing:**
   [Detailed information]
   
   **Duration:** [Specific]
   
   **Rationale:**
   [Explanation]
   
   **Contraindications:**
   [When not to use]
   
   **Monitoring:**
   [What to watch]
   
   ---
   **Reference:** WHO AWaRe Classification of Antibiotics, 2021
   **Pages:** [specific pages]
   ```

6. **CRITICAL RULES:**
   - Stick to what's in the document
   - Be complete for what's asked
   - Be focused - don't add unnecessary sections
   - Include specific clinical details
   - Always specify AWaRe group
   - Always cite with page numbers
   - Never include GitHub or web links in citations

7. **EXAMPLES:**

   Question: "First-line for pneumonia?"
   ✅ Provide detailed first-line info ONLY + citation with pages
   ❌ Don't include second-line unless asked
   
   Question: "What are alternatives?"
   ✅ Provide alternatives + citation with pages
   
   Question: "Complete protocol?"
   ✅ Provide first, second, reserve + citation with pages

Remember: Professional consultation required for actual treatment decisions."""
        }
    ]

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "show_info" not in st.session_state:
    st.session_state.show_info = True

# ================================
# MAIN LAYOUT
# ================================

create_header()
create_sidebar()

# Show info screen
if st.session_state.show_info:
    col1, col2 = st.columns([4, 1])
    with col1:
        show_description()
        show_source_info(pdf_loaded)
        show_disclaimer()
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("✅ I Understand\n\nStart →", use_container_width=True):
            st.session_state.show_info = False
            st.rerun()
    st.stop()

# ================================
# CHAT INTERFACE
# ================================

st.markdown("### 💬 Chat with WHO AWaRe Bot")
col1, col2 = st.columns(2)
with col1:
    if st.button("📖 View Info"):
        st.session_state.show_info = True
        st.rerun()
with col2:
    if st.button("🔄 New Chat"):
        st.session_state.display_messages = []
        st.rerun()

st.markdown("---")

# Welcome message
if not st.session_state.display_messages:
    st.info("👋 Ask about antibiotics and I'll provide focused, detailed information with proper guideline citations!")

# Display messages
for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 Ask about antibiotics based on WHO AWaRe classification..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages,
                stream=True,
                temperature=0.7,
                max_tokens=2500
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.error(error_message)
            full_response = error_message
    
    st.session_state.display_messages.append({"role": "assistant", "content": full_response})
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>WHO AWaRe Antibiotics Chatbot v3.2</strong></p>
    <p style="font-size: 0.9em;">Based exclusively on WHO AWaRe Classification Document</p>
    <p style="font-size: 0.85em; margin-top: 10px;">
        ⚕️ For educational purposes only • Always consult healthcare professionals
    </p>
</div>
""", unsafe_allow_html=True)
