# Study AI — RAG-Powered Study Notes Generator (Multi-Provider)
# Supports: Ollama (Local), Google Gemini (API), Groq (API)


import streamlit as st
from rag_engine import (
    extract_text,
    chunk_text,
    index_documents,
    query_documents,
    clear_collection,
    get_indexed_sources,
    get_or_create_collection,
    build_rag_prompt,
)
from llm_provider import (
    get_available_providers,
    generate,
)

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Study AI — RAG",
    page_icon="🤖",
    layout="centered"
)

# -------------------------
# Custom Styling (Glassmorphism & Dark Mode)
# -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Font Override */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Background styling: Deep Matte Black with dark grey glow */
    [data-testid="stAppViewContainer"] {
        background-color: #0c0c0c !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(82, 82, 91, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(38, 38, 38, 0.15) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
    }

    /* Header transparent glass styling */
    [data-testid="stHeader"] {
        background: rgba(12, 12, 12, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Sidebar glassmorphic styling */
    [data-testid="stSidebar"] {
        background: rgba(23, 23, 23, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #d4d4d4 !important;
    }

    /* Custom Gradient Title (Charcoal to Silver/White) */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #ffffff 0%, #a3a3a3 60%, #525252 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0px !important;
    }

    /* Subheadings styling */
    h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #f5f5f5 !important;
    }

    /* Caption styling */
    div[data-testid="stCaptionContainer"] {
        color: #737373 !important;
        font-size: 14px !important;
        margin-top: -8px !important;
        margin-bottom: 24px !important;
    }

    /* Text inputs */
    div[data-testid="stTextInput"] label {
        font-weight: 500 !important;
        color: #a3a3a3 !important;
        font-size: 15px !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: rgba(38, 38, 38, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        color: #e5e5e5 !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(8px) !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #737373 !important;
        box-shadow: 0 0 0 3px rgba(115, 115, 115, 0.25) !important;
        background-color: rgba(38, 38, 38, 0.6) !important;
    }

    /* Button: Generate Notes (Black/Charcoal with Silver glow on hover) */
    div[data-testid="stButton"] button {
        background: #171717 !important;
        border: 1px solid #404040 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.5) !important;
        width: 100% !important;
    }

    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        background: #262626 !important;
        border-color: #737373 !important;
        box-shadow: 0 6px 20px 0 rgba(255, 255, 255, 0.05) !important;
    }

    div[data-testid="stButton"] button:active {
        transform: translateY(0) !important;
    }

    /* Button: Download Notes */
    div[data-testid="stDownloadButton"] button {
        background: rgba(23, 23, 23, 0.3) !important;
        border: 1px solid rgba(163, 163, 163, 0.4) !important;
        color: #e5e5e5 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        backdrop-filter: blur(8px) !important;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: #ffffff !important;
        box-shadow: 0 4px 12px 0 rgba(255, 255, 255, 0.05) !important;
        transform: translateY(-1px) !important;
    }

    /* Container for generated notes */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(23, 23, 23, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-top: 24px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6) !important;
    }

    /* Sidebar Divider & list tweaks */
    hr {
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Code blocks formatting */
    code {
        background: rgba(23, 23, 23, 0.8) !important;
        color: #e5e5e5 !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
        font-family: monospace !important;
        font-size: 14px !important;
    }
    
    pre code {
        display: block !important;
        padding: 12px !important;
        overflow-x: auto !important;
    }

    /* Tab styling */
    div[data-testid="stTabs"] button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        color: #a3a3a3 !important;
        font-size: 15px !important;
        padding: 10px 20px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
    }

    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        background: rgba(23, 23, 23, 0.3) !important;
        border: 1px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    div[data-testid="stFileUploader"] label {
        color: #a3a3a3 !important;
        font-weight: 500 !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: rgba(23, 23, 23, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] summary {
        color: #d4d4d4 !important;
        font-weight: 500 !important;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background: rgba(23, 23, 23, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    div[data-testid="stMetric"] label {
        color: #a3a3a3 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Select box styling */
    div[data-testid="stSelectbox"] label {
        color: #a3a3a3 !important;
        font-weight: 500 !important;
    }

    /* Radio button styling */
    div[data-testid="stRadio"] label {
        color: #a3a3a3 !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Header with SVG Robot Logo
# -------------------------
st.markdown(
    """
<div style="display: flex; align-items: center; gap: 18px; margin-bottom: 28px; margin-top: 10px;">
<svg width="58" height="58" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 12px rgba(255,255,255,0.18));">
<!-- Antenna -->
<rect x="11.25" y="1" width="1.5" height="4" rx="0.75" fill="#a3a3a3"/>
<circle cx="12" cy="1.5" r="1.5" fill="#ffffff"/>
<!-- Head -->
<rect x="4" y="5" width="16" height="12" rx="3" fill="#171717" stroke="#404040" stroke-width="1.5"/>
<!-- Eyes -->
<circle cx="8.5" cy="10.5" r="1.5" fill="#ffffff"/>
<circle cx="15.5" cy="10.5" r="1.5" fill="#ffffff"/>
<!-- Ears -->
<rect x="2.5" y="9" width="1.5" height="4" rx="0.75" fill="#404040"/>
<rect x="20" y="9" width="1.5" height="4" rx="0.75" fill="#404040"/>
<!-- Mouth -->
<path d="M8 14H16" stroke="#a3a3a3" stroke-width="1.5" stroke-linecap="round"/>
<!-- Neck -->
<rect x="10" y="17" width="4" height="1.5" fill="#262626"/>
<!-- Body -->
<path d="M6 18.5H18L19 23H5L6 18.5Z" fill="#171717" stroke="#404040" stroke-width="1.5"/>
</svg>
<div>
<h1 style="margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 2.6rem; line-height: 1.1; background: linear-gradient(135deg, #ffffff 0%, #a3a3a3 60%, #525252 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none !important; box-shadow: none !important; box-decoration-break: clone !important;">Study AI</h1>
<p style="margin: 2px 0 0 0; color: #737373; font-size: 13.5px; font-weight: 500; font-family: 'Outfit', sans-serif; letter-spacing: 0.5px;">RAG-Powered • Multi-Provider AI</p>
</div>
</div>
""",
    unsafe_allow_html=True
)

# -------------------------
# Initialize Session State
# -------------------------
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0
if "rag_answer" not in st.session_state:
    st.session_state.rag_answer = None
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = None
if "general_notes" not in st.session_state:
    st.session_state.general_notes = None

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("📖 About")
    st.write(
        """
        **RAG-powered** study notes generator.
        Upload your PDFs or text files, then ask 
        questions grounded in **your** documents.
        """
    )

    st.markdown("---")

    # AI Provider Selection
    st.write("### 🤖 AI Provider")
    providers = get_available_providers()
    provider_names = list(providers.keys())
    provider_labels = []
    for name, info in providers.items():
        icon = info["icon"]
        status = "✅" if info["available"] else "🔑 No key"
        provider_labels.append(f"{icon} {name} {status}")

    selected_idx = st.selectbox(
        "Choose AI Provider",
        range(len(provider_names)),
        format_func=lambda i: provider_labels[i],
        index=0,
        key="provider_select",
    )
    selected_provider = provider_names[selected_idx]
    provider_info = providers[selected_provider]

    # Model selection for the chosen provider
    selected_model = st.selectbox(
        "Model",
        provider_info["models"],
        key="model_select",
    )

    if not provider_info["available"]:
        st.warning(f"⚠️ Add `{provider_info['env_key']}` to your `.env` file to use {selected_provider}")

    st.markdown("---")

    # Document Upload Section
    st.write("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your study materials here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or TXT files to build your knowledge base",
    )

    if uploaded_files:
        if st.button("📥 Index Documents", use_container_width=True):
            collection = get_or_create_collection()
            total_chunks = 0
            progress_bar = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        text = extract_text(file, file.name)

                        if not text.strip():
                            st.warning(f"⚠️ No text found in {file.name}")
                            continue

                        chunks = chunk_text(text)
                        num_indexed = index_documents(chunks, file.name, collection)
                        total_chunks += num_indexed

                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")

                progress_bar.progress((idx + 1) / len(uploaded_files))

            if total_chunks > 0:
                st.success(f"✅ Indexed **{total_chunks}** chunks from **{len(uploaded_files)}** file(s)")
                st.session_state.indexed_count = total_chunks

            progress_bar.empty()

    st.markdown("---")

    # Knowledge Base Status
    st.write("### 🗄️ Knowledge Base")
    try:
        sources = get_indexed_sources()
        if sources:
            total = sum(sources.values())
            st.metric("Total Chunks", total)
            st.metric("Documents", len(sources))

            with st.expander("📑 Indexed Documents"):
                for source, count in sources.items():
                    st.write(f"• **{source}** — {count} chunks")

            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                clear_collection()
                st.session_state.indexed_count = 0
                st.session_state.rag_answer = None
                st.session_state.rag_sources = None
                st.success("Knowledge base cleared!")
                st.rerun()
        else:
            st.info("No documents indexed yet. Upload files above to get started.")
    except Exception:
        st.info("No documents indexed yet. Upload files above to get started.")

    st.markdown("---")

    st.write("### ✨ Features")
    st.write("✅ RAG-powered answers")
    st.write("✅ Multi-provider AI")
    st.write("✅ PDF & TXT support")
    st.write("✅ Source citations")
    st.write("✅ Download notes")

# -------------------------
# Main Area — Two Modes
# -------------------------
tab_rag, tab_general = st.tabs(["📄 Ask Your Documents", "💡 General Notes"])

# -------------------------
# TAB 1: RAG Mode
# -------------------------
with tab_rag:
    st.markdown(
        f"""
        <p style="color: #a3a3a3; font-size: 14px; margin-bottom: 20px;">
        Ask questions about your uploaded documents. Using <strong>{provider_info['icon']} {selected_provider}</strong> → <strong>{selected_model}</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Ask a question about your documents",
        placeholder="Example: What are the key concepts in Chapter 3?",
        key="rag_question",
    )

    if st.button("🔍 Ask Your Documents", use_container_width=True, key="rag_btn"):

        if not question or question.strip() == "":
            st.warning("Please enter a question.")
        elif not provider_info["available"]:
            st.error(f"❌ {selected_provider} requires an API key. Add it to your `.env` file.")
        else:
            try:
                sources = get_indexed_sources()
                if not sources:
                    st.warning("⚠️ No documents indexed yet. Upload and index files in the sidebar first.")
                else:
                    with st.spinner(f"Searching docs & generating with {selected_provider}..."):
                        # Step 1: Retrieve relevant chunks
                        retrieved = query_documents(question)

                        if not retrieved:
                            st.warning("No relevant content found in your documents.")
                        else:
                            # Step 2: Build RAG prompt
                            rag_prompt = build_rag_prompt(question, retrieved)

                            # Step 3: Generate with selected provider
                            answer = generate(rag_prompt, selected_provider, selected_model)

                            st.session_state.rag_answer = answer
                            st.session_state.rag_sources = retrieved

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display RAG results
    if st.session_state.rag_answer:
        st.success("Answer generated from your documents!")

        st.subheader("📚 Answer")
        with st.container(border=True):
            st.markdown(st.session_state.rag_answer)

        if st.session_state.rag_sources:
            with st.expander("📑 Source Chunks Used", expanded=False):
                for i, chunk in enumerate(st.session_state.rag_sources, 1):
                    st.markdown(
                        f"""
                        <div style="background: rgba(38, 38, 38, 0.5); border: 1px solid rgba(255,255,255,0.06); 
                        border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                        <p style="color: #737373; font-size: 12px; margin: 0 0 6px 0;">
                        🔖 Source: <strong>{chunk['source']}</strong> • Relevance: {1 - chunk['distance']:.0%}
                        </p>
                        <p style="color: #d4d4d4; font-size: 14px; margin: 0; line-height: 1.6;">
                        {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.download_button(
            label="📥 Download Answer",
            data=st.session_state.rag_answer,
            file_name="rag_study_notes.txt",
            mime="text/plain",
            use_container_width=True,
        )

# -------------------------
# TAB 2: General Notes Mode
# -------------------------
with tab_general:
    st.markdown(
        f"""
        <p style="color: #a3a3a3; font-size: 14px; margin-bottom: 20px;">
        Generate general study notes on any topic. Using <strong>{provider_info['icon']} {selected_provider}</strong> → <strong>{selected_model}</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "Enter a topic",
        placeholder="Example: Machine Learning",
        key="general_topic",
    )

    if st.button("Generate Notes", use_container_width=True, key="general_btn"):

        if not topic or topic.strip() == "":
            st.warning("Please enter a topic.")
        elif not provider_info["available"]:
            st.error(f"❌ {selected_provider} requires an API key. Add it to your `.env` file.")
        else:
            with st.spinner(f"Generating with {selected_provider}..."):
                try:
                    prompt = f"""
Write concise study notes about {topic}.

Format:
1. Definition
2. Key Points
3. Examples
4. Summary

Keep the notes easy to understand.
"""
                    st.session_state.general_notes = generate(prompt, selected_provider, selected_model)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    if st.session_state.general_notes:
        st.success("Notes generated successfully!")

        st.subheader("📚 Study Notes")
        with st.container(border=True):
            st.markdown(st.session_state.general_notes)

        st.download_button(
            label="📥 Download Notes",
            data=st.session_state.general_notes,
            file_name="study_notes.txt",
            mime="text/plain",
            use_container_width=True,
        )

# -------------------------
# Footer
# -------------------------
st.divider()

st.caption(
    "Built using Streamlit, ChromaDB & Multi-Provider AI — Ollama • Gemini • Groq"
)
