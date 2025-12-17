"""
WHO Antibiotics Use Support Chatbot - WITH CITATIONS & TREATMENT HIERARCHY
A professional Streamlit chatbot for WHO antibiotic resistance guidelines
Version: 2.1 - Enhanced with Citations
"""

import streamlit as st
from openai import OpenAI
import base64
from pathlib import Path

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="WHO Antibiotics Support Bot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.who.int/health-topics/antimicrobial-resistance',
        'Report a bug': None,
        'About': "WHO Antibiotics Support Chatbot - Powered by OpenAI"
    }
)

# ================================
# LOGO HELPER FUNCTION
# ================================
def get_logo_base64():
    """Convert logo to base64 for embedding"""
    logo_path = Path("logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    return None

# ================================
# CUSTOM CSS STYLING
# ================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        padding-top: 2rem;
    }
    
    /* Header container */
    .header-container {
        background: linear-gradient(135deg, #0051A5 0%, #00A3DD 100%);
        padding: 40px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .logo-img {
        width: 150px;
        height: 150px;
        margin: 0 auto 20px;
        display: block;
        border-radius: 50%;
        border: 5px solid white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    .header-title {
        color: white;
        font-size: 2.8em;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #E0F2FF;
        font-size: 1.3em;
        margin-top: 10px;
        font-weight: 300;
    }
    
    /* Description box */
    .description-box {
        background-color: #F8F9FA;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #0051A5;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .description-box h3 {
        color: #0051A5;
        margin-top: 0;
        font-size: 1.5em;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background-color: #FFF3CD;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #FFC107;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .disclaimer-box h3 {
        color: #856404;
        margin-top: 0;
        font-size: 1.5em;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        color: #0051A5;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0051A5 0%, #00A3DD 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,81,165,0.4);
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 30px;
        margin-top: 50px;
        border-top: 2px solid #E0E0E0;
        background-color: #F8F9FA;
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F0F2F6;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0051A5;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER WITH LOGO
# ================================
def create_header():
    """Create professional header with logo and title"""
    
    logo_base64 = get_logo_base64()
    
    if logo_base64:
        st.markdown(f"""
        <div class="header-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="WHO Logo">
            <h1 class="header-title">WHO Antibiotics Support Chatbot</h1>
            <p class="header-subtitle">Evidence-Based Guidance with Citations & Treatment Hierarchy</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="header-container">
            <div style="font-size: 4em; margin-bottom: 10px;">💊 🏥 🌍</div>
            <h1 class="header-title">WHO Antibiotics Support Chatbot</h1>
            <p class="header-subtitle">Evidence-Based Guidance with Citations & Treatment Hierarchy</p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# DESCRIPTION SECTION
# ================================
def show_description():
    """Display chatbot description and features"""
    
    st.markdown("""
    <div class="description-box">
        <h3>📖 About This Chatbot</h3>
        <p style="font-size: 1.1em; line-height: 1.8;">
            This AI-powered assistant provides comprehensive, evidence-based information on antibiotic use 
            and antimicrobial resistance based on <strong>World Health Organization (WHO)</strong> guidelines.
        </p>
        <h4 style="color: #0051A5; margin-top: 20px;">Enhanced Features:</h4>
        <ul style="font-size: 1.05em; line-height: 2;">
            <li>🔹 <strong>Clear Treatment Hierarchy</strong> - First-line, second-line, and alternative therapies clearly identified</li>
            <li>🔹 <strong>Cited References</strong> - All responses include WHO guideline citations</li>
            <li>🔹 <strong>WHO AWaRe Classification</strong> - Access, Watch, and Reserve categories specified</li>
            <li>🔹 <strong>Age-Specific Dosing</strong> - Pediatric and adult recommendations separated</li>
            <li>🔹 <strong>Resistance Patterns</strong> - Local and global antimicrobial resistance data</li>
            <li>🔹 <strong>Safety Guidelines</strong> - Contraindications, adverse effects, and monitoring</li>
        </ul>
        <div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <p style="margin: 0; color: #1565C0; font-weight: 600;">
                📚 <strong>Citation Format:</strong> Every response includes WHO references and publication details at the end.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================
# DISCLAIMER SECTION
# ================================
def show_disclaimer():
    """Display medical disclaimer"""
    
    st.markdown("""
    <div class="disclaimer-box">
        <h3>⚠️ Important Medical Disclaimer</h3>
        <p style="font-size: 1.1em; line-height: 1.6; color: #856404; font-weight: 600;">
            This chatbot is for <strong>INFORMATIONAL and EDUCATIONAL purposes ONLY</strong>
        </p>
        
        <h4 style="color: #856404; margin-top: 20px;">❌ This Chatbot Does NOT:</h4>
        <ul style="font-size: 1em; line-height: 1.8; color: #856404;">
            <li>Replace professional medical advice, diagnosis, or treatment</li>
            <li>Provide patient-specific medical recommendations</li>
            <li>Authorize prescription or medication changes</li>
            <li>Offer emergency medical guidance</li>
        </ul>
        
        <h4 style="color: #856404; margin-top: 20px;">✅ You MUST:</h4>
        <ul style="font-size: 1em; line-height: 1.8; color: #856404;">
            <li><strong>Always consult qualified healthcare professionals</strong> for medical decisions</li>
            <li><strong>Seek immediate medical attention</strong> for emergencies</li>
            <li><strong>Follow your doctor's prescriptions</strong> and treatment plans</li>
            <li><strong>Consider local resistance patterns</strong> and antibiograms</li>
        </ul>
        
        <div style="background-color: #FFF; padding: 15px; border-radius: 8px; margin-top: 20px; border: 2px solid #856404;">
            <p style="margin: 0; color: #856404; font-weight: 700; text-align: center;">
                ⚕️ By using this chatbot, you acknowledge that you understand this disclaimer 
                and will consult healthcare professionals for medical advice.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================
# SIDEBAR CONTENT
# ================================
def create_sidebar():
    """Create informative sidebar"""
    
    with st.sidebar:
        # Logo in sidebar
        logo_base64 = get_logo_base64()
        if logo_base64:
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{logo_base64}" style="width: 120px; border-radius: 50%; border: 3px solid #0051A5;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; font-size: 3em; margin: 20px 0;">
                💊
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<p class="sidebar-header">Quick Navigation</p>', unsafe_allow_html=True)
        
        # Information sections
        with st.expander("📚 How to Use This Bot", expanded=True):
            st.markdown("""
            **Getting Started:**
            1. 💬 Type your question in the chat box
            2. 📋 Receive structured responses with:
               - First-line treatments
               - Second-line alternatives
               - WHO guidelines citations
            
            **Example Questions:**
            - *"First-line treatment for pneumonia in children?"*
            - *"What are alternatives to penicillin?"*
            - *"WHO recommendations for UTI treatment?"*
            """)
        
        with st.expander("🎯 Response Format"):
            st.markdown("""
            **Each response includes:**
            
            ✅ **Treatment Hierarchy**
            - 🟢 First-line options
            - 🟡 Second-line alternatives
            - 🔴 Reserve antibiotics (if applicable)
            
            ✅ **Detailed Information**
            - Dosing (adult & pediatric)
            - Duration of treatment
            - Contraindications
            - Adverse effects
            
            ✅ **Citations**
            - WHO guideline references
            - Publication dates
            - AWaRe classification
            """)
        
        with st.expander("🌐 WHO AWaRe Classification"):
            st.markdown("""
            **🟢 ACCESS Group (First-line):**
            - Narrow-spectrum antibiotics
            - Lower resistance risk
            - Examples: Amoxicillin, Penicillin
            
            **🟡 WATCH Group (Second-line):**
            - Broader spectrum antibiotics
            - Higher resistance potential
            - Examples: Ciprofloxacin, Ceftriaxone
            
            **🔴 RESERVE Group (Last-resort):**
            - Reserved for specific cases
            - Highest resistance concern
            - Examples: Colistin, Linezolid
            """)
        
        with st.expander("📖 Citation Examples"):
            st.markdown("""
            **Responses will cite:**
            
            1. WHO Model List of Essential Medicines
            2. WHO AWaRe Classification (2021)
            3. WHO Guidelines for Management of Common Childhood Illnesses
            4. WHO Pocket Book of Hospital Care for Children
            5. WHO Guidelines on Tuberculosis
            6. Specific condition guidelines
            
            *All with publication year and edition*
            """)
        
        st.divider()
        
        # Statistics
        st.markdown("### 📊 Session Statistics")
        if "messages" in st.session_state:
            message_count = len([m for m in st.session_state.messages if m["role"] != "system"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", message_count)
            with col2:
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                st.metric("Questions", user_msgs)
        else:
            st.metric("Total Messages", 0)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.display_messages = []
            st.session_state.show_info = True
            st.rerun()
        
        st.divider()
        
        # Footer links
        st.markdown("""
        ### 🔗 WHO Resources
        - [WHO AMR Portal](https://www.who.int/health-topics/antimicrobial-resistance)
        - [AWaRe Database](https://www.who.int/publications/i/item/2021-aware-classification)
        - [Essential Medicines List](https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines)
        - [Pediatric Guidelines](https://www.who.int/publications/i/item/9789241548373)
        """)
        
        st.markdown("---")
        st.caption("💊 Version 2.1 with Citations")

# ================================
# API KEY CHECK
# ================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ **OpenAI API key not found!**")
    st.info("""
    **Configuration Required:**
    
    **For Local Development:**
    1. Create `.streamlit/secrets.toml` file
    2. Add: `OPENAI_API_KEY = "your-api-key-here"`
    
    **For Streamlit Cloud:**
    1. Go to App Settings → Secrets
    2. Add your API key in TOML format
    """)
    st.stop()

# ================================
# INITIALIZE OPENAI CLIENT
# ================================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"❌ Error initializing OpenAI: {str(e)}")
    st.stop()

# ================================
# ENHANCED SYSTEM PROMPT
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """You are an expert medical information assistant specializing in WHO (World Health Organization) antibiotic resistance guidelines and antimicrobial stewardship.

CRITICAL RESPONSE REQUIREMENTS:

1. **TREATMENT HIERARCHY - ALWAYS CLEARLY IDENTIFY:**

   **FIRST-LINE TREATMENT:**
   - Clearly label as "FIRST-LINE" or "Primary Treatment"
   - Specify the preferred antibiotic(s)
   - Include dosing, duration, and WHO AWaRe classification
   - Explain why it's the first choice
   
   **SECOND-LINE TREATMENT:**
   - Clearly label as "SECOND-LINE" or "Alternative Treatment"
   - Specify when to use (treatment failure, allergies, contraindications)
   - Include dosing, duration, and WHO AWaRe classification
   - List specific scenarios for use
   
   **RESERVE ANTIBIOTICS (if applicable):**
   - Clearly label as "RESERVE" or "Last-Resort"
   - Explain limited use cases
   - Emphasize resistance concerns

2. **CITATIONS - MANDATORY AT END OF EVERY RESPONSE:**

   Always end your response with a "References & Guidelines" section containing:
   - Specific WHO guideline titles
   - Publication years
   - Edition numbers when applicable
   - Relevant WHO AWaRe classification version
   - Page numbers or section references when available
   
   Example format:
   ---
   ## 📚 References & Guidelines
   
   1. WHO Model List of Essential Medicines, 22nd Edition (2021)
   2. WHO AWaRe Classification of Antibiotics for Evaluation and Monitoring of Use (2021)
   3. WHO Recommendations on the Management of Diarrhea and Pneumonia in HIV-infected Infants and Children (2010)
   4. WHO Pocket Book of Hospital Care for Children, 2nd Edition (2013)
   
   **Note:** Treatment decisions should be based on local antibiograms and resistance patterns in consultation with healthcare professionals.

3. **STRUCTURE YOUR RESPONSES:**

   Use this format:
   
   ## [Condition Name] - Treatment Guidelines
   
   ### 🟢 FIRST-LINE TREATMENT
   **Antibiotic Name** (AWaRe Classification: Access/Watch/Reserve)
   - Dosing: [specific doses]
   - Duration: [specific duration]
   - Rationale: [why first-line]
   
   ### 🟡 SECOND-LINE TREATMENT
   **Alternative Options:**
   **Antibiotic Name** (AWaRe Classification)
   - When to use: [specific scenarios]
   - Dosing: [specific doses]
   - Duration: [specific duration]
   
   ### 📋 Additional Considerations
   - Age-specific modifications
   - Contraindications
   - Adverse effects
   - Monitoring requirements
   
   ### 📚 References & Guidelines
   [Citations as specified above]

4. **COMPREHENSIVE DETAILS:**
   - Include specific dosages (mg/kg for pediatrics, fixed doses for adults)
   - Specify treatment duration
   - Mention WHO AWaRe classification for each antibiotic
   - Address age-specific considerations
   - Include contraindications and safety information
   - Discuss local resistance pattern considerations

5. **EMPHASIZE ANTIMICROBIAL STEWARDSHIP:**
   - Explain rationale for treatment choices
   - Highlight importance of completing courses
   - Mention resistance prevention
   - Stress professional consultation

Always maintain scientific accuracy and emphasize that users must consult healthcare professionals for specific medical advice and treatment decisions. Your goal is to educate healthcare professionals and students with properly cited, evidence-based information."""
        }
    ]

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "show_info" not in st.session_state:
    st.session_state.show_info = True

# ================================
# MAIN LAYOUT
# ================================

# Create header
create_header()

# Create sidebar
create_sidebar()

# Show description and disclaimer on first load
if st.session_state.show_info:
    col1, col2 = st.columns([4, 1])
    with col1:
        show_description()
        show_disclaimer()
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("✅ I Understand\n\nStart Chatting →", use_container_width=True):
            st.session_state.show_info = False
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 View Info Again", use_container_width=True):
            st.session_state.show_info = True
            st.rerun()
    
    st.stop()

# ================================
# CHAT INTERFACE
# ================================

# Quick action buttons above chat
st.markdown("### 💬 Chat Interface")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📖 View Info Again"):
        st.session_state.show_info = True
        st.rerun()
with col2:
    if st.button("🔄 New Conversation"):
        st.session_state.display_messages = []
        st.rerun()

st.markdown("---")

# Display chat messages
if not st.session_state.display_messages:
    st.info("👋 Welcome! Ask about treatment guidelines and I'll provide first-line and second-line options with WHO citations.")

for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 Ask about antibiotics, treatment guidelines, or WHO recommendations..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to display history
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    
    # Add to full chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Create streaming response
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages,
                stream=True,
                temperature=0.7,
                max_tokens=2500
            )
            
            # Stream the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Display final response
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}\n\nPlease try again or contact support if the issue persists."
            message_placeholder.error(error_message)
            full_response = error_message
    
    # Add assistant response to histories
    st.session_state.display_messages.append({"role": "assistant", "content": full_response})
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3 style="color: #0051A5; margin-bottom: 15px;">💊 WHO Antibiotics Support Chatbot</h3>
    <p style="font-size: 1.05em; margin-bottom: 10px;">
        <strong>Version 2.1</strong> - Enhanced with Citations & Treatment Hierarchy
    </p>
    <p style="font-size: 1.05em; margin-bottom: 10px;">
        <strong>Powered by OpenAI GPT-3.5</strong> | Based on WHO Guidelines
    </p>
    <p style="font-size: 0.95em; color: #666; margin-bottom: 20px;">
        All responses include WHO guideline citations and clear treatment hierarchies
    </p>
    <p style="font-size: 0.9em; color: #999;">
        ⚕️ <strong>Medical Disclaimer:</strong> Always consult healthcare professionals for medical advice.<br>
        This chatbot provides educational information with proper citations for reference only.
    </p>
    <p style="font-size: 0.85em; color: #999; margin-top: 15px;">
        © 2024 WHO Antibiotics Support Bot | For Educational Purposes
    </p>
</div>
""", unsafe_allow_html=True)
