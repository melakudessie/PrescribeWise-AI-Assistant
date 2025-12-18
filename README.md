# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

# 🩺 PrescribeWise - Health Worker Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://raise4impact.streamlit.app/)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)

**PrescribeWise** is an AI-powered clinical decision support tool designed to help health workers, pharmacists, and clinicians instant access to the **WHO AWaRe (Access, Watch, Reserve) Antibiotic Book**.

By utilizing **Retrieval-Augmented Generation (RAG)**, this assistant synthesizes over 700 pages of global medical guidelines into accurate, cited, and color-coded treatment recommendations.

🔗 **Live Demo:** [https://raise4impact.streamlit.app/](https://raise4impact.streamlit.app/)

---

## 🚀 Key Features

### 1. 🧠 Evidence-Based Answers (RAG)
Unlike standard chatbots, PrescribeWise does not hallucinate medical advice. It answers **strictly** based on the provided WHO Guidelines (`WHOAMR.pdf`). If the information is not in the text, it will not invent an answer.

### 2. 🚦 AWaRe Color-Coding
The app automatically formats antibiotic recommendations using the WHO stewardship classification for instant visual recognition:
* :green_circle: **:green[Access (Green)]:** First-choice antibiotics that are generally safe and widely available.
* :orange_circle: **:orange[Watch (Orange)]:** Antibiotics with higher resistance potential, used as second choices.
* :red_circle: **:red[Reserve (Red)]:** "Last resort" antibiotics saved for multidrug-resistant infections.

### 3. 📍 Precision Citations
Every clinical claim is backed by a specific source reference.
> *Example Output:* "The first-line treatment for acute otitis media is Amoxicillin... **[Page 45]**"

### 4. 🔍 Clinical Evidence Viewer
For full transparency, users can expand the **"View Clinical Evidence"** section to read the exact raw text chunks retrieved from the PDF that were used to generate the answer.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/) (Modern LCEL Architecture)
* **LLM:** OpenAI GPT-4 (Temperature set to 0.0 for maximum factuality)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** OpenAI Embeddings
* **Document Parsing:** PyPDF & RecursiveCharacterTextSplitter

---

## ⚙️ Installation & Local Setup

Follow these steps to run the application on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/melakudessie/PrescribeWise-AI-Assistant.git](https://github.com/melakudessie/PrescribeWise-AI-Assistant.git)
cd PrescribeWise-AI-Assistant
