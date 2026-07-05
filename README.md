# 🤖 AI-Study Notes Generator (RAG-Powered)

<p align="center">
  <img src="screenshots/Home_screen.png" alt="Study AI Banner" width="100%" style="border: 2px solid #1a1a1a; border-radius: 8px; box-shadow: 4px 4px 0px 0px #1a1a1a;"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Powered%20By-Ollama%20%2B%20Gemma%203-lightgrey?style=for-the-badge&logo=cpu&logoColor=white&color=262626" alt="Ollama + Gemma 3"/>
  <img src="https://img.shields.io/badge/RAG-ChromaDB%20%2B%20Nomic%20Embed-blue?style=for-the-badge&logo=database&logoColor=white&color=1a1a2e" alt="RAG Pipeline"/>
  <img src="https://img.shields.io/badge/UI-Custom%20Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white&color=171717" alt="Custom Streamlit UI"/>
  <img src="https://img.shields.io/badge/Privacy-100%25%20Local%20%26%20Private-green?style=for-the-badge&logo=shield&logoColor=white&color=404040" alt="Privacy First"/>
</p>

---

## 📖 Introduction

**Study AI** is a **RAG-powered** (Retrieval-Augmented Generation) study notes generator. Upload your own PDFs and text files, and get AI-generated answers **grounded in your actual documents** — not just LLM training data.

Built with **Ollama**, **Gemma 3**, **ChromaDB**, and a custom **glassmorphic matte-black** Streamlit interface. Everything runs **100% locally** on your machine.

---

## ✨ Key Features

*   **📄 RAG Pipeline**: Upload PDFs/TXT files → documents are chunked, embedded, and stored → ask questions → get grounded answers with source citations
*   **🔒 Local-First Privacy**: All processing happens on your machine using Ollama. No API keys, no cloud services.
*   **🎨 Premium Glassmorphism UI**: Monochromatic dark-mode styling with frosted-glass containers, glowing borders, and silver gradients.
*   **📑 Source Citations**: Every answer shows which source chunks were used, with relevance scores.
*   **🔄 Two Modes**: 
    - **RAG Mode**: Ask questions about your uploaded documents
    - **General Mode**: Generate study notes on any topic from LLM knowledge
*   **📥 One-Click Export**: Download generated notes as text files.
*   **🗄️ Persistent Knowledge Base**: Indexed documents persist between sessions via ChromaDB.

---

## 🧩 Technical Architecture

### RAG Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload PDF  │ ──▶ │  Extract     │ ──▶ │  Chunk Text  │ ──▶ │  Embed with  │
│  or TXT      │     │  Text        │     │  (500 chars) │     │  nomic-embed │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Grounded    │ ◀── │  Generate    │ ◀── │  Inject top  │ ◀── │  Store in    │
│  Answer +    │     │  with Gemma  │     │  5 chunks    │     │  ChromaDB    │
│  Sources     │     │  3 via Ollama│     │  into prompt │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                ▲
                                                │
                                          ┌─────┴──────┐
                                          │ User asks  │
                                          │ a question │
                                          └────────────┘
```

### Component Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM | Ollama (Gemma 3 1B) | Answer generation |
| Embeddings | Ollama (nomic-embed-text) | Document & query embedding |
| Vector DB | ChromaDB | Persistent vector storage & retrieval |
| PDF Parser | PyPDF2 | PDF text extraction |
| Chunking | LangChain Text Splitters | Intelligent recursive text splitting |
| UI | Streamlit | Web dashboard |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Ollama** installed.

*   Download Ollama: [ollama.com/download](https://ollama.com/download)
*   Pull the required models:
    ```bash
    ollama pull gemma3:1b
    ollama pull nomic-embed-text
    ```

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/vedantdubey19/AI-Study-Notes-Generator.git
cd AI-Study-Notes-Generator

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py --browser.gatherUsageStats false
```
The application will launch at **`http://localhost:8501`**.

---

## 📂 Project Structure

```
AI-Study-Notes-Generator/
├── screenshots/               # Application UI screenshots
│   ├── Home_screen.png
│   ├── Generating_Notes.png
│   └── Results.png
├── chroma_db/                 # Persistent vector database (auto-created)
├── streamlit_app.py           # Main Streamlit app with RAG UI
├── rag_engine.py              # RAG backend: extraction, chunking, embedding, retrieval
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Excluded files
```

---

## 📸 How to Use

1. **Upload Documents**: Drag & drop PDF or TXT files in the sidebar
2. **Index**: Click "Index Documents" to process and embed your files
3. **Ask Questions**: Switch to the "Ask Your Documents" tab, type your question
4. **Get Grounded Answers**: Receive answers based on your actual documents, with source citations
5. **General Mode**: Use the "General Notes" tab for topic-based notes from LLM knowledge

---

## 👥 Author

*   **Vedant Dubey** - [@vedantdubey19](https://github.com/vedantdubey19)

---
## Collaborative Update 
Improved project documentatio and project details
