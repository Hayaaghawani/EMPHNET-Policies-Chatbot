"""
Streamlit Web Application for EMPHNET HR Policy Chatbot

Simple, minimal interface for staff to ask HR policy questions and get grounded answers.
"""

import streamlit as st
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG modules
from src.retrieval import HybridRetriever, load_chunks_from_json
from src.generation import OllamaGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="EMPHNET HR Policy Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .question-box {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            color: #1E1E1E;
        }
        .answer-box {
            background-color: #e8f5e9;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #4caf50;
            color: #1E1E1E;
        }
        .source-box {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
            border-left: 4px solid #ff9800;
            color: #1E1E1E;
        }
        .confidence-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.85rem;
            font-weight: bold;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize retriever and generator (cached for performance)"""
    try:
        # Get configuration from environment
        chroma_db_path = os.getenv("CHROMA_DB_PATH", "chroma_storage")
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        top_k = int(os.getenv("TOP_K_RESULTS", "8"))
        
        # Initialize retriever
        st.status("🔧 Initializing Retriever...", expanded=False)
        retriever = HybridRetriever(
            chroma_db_path=chroma_db_path,
            embedding_model=embedding_model,
            top_k=top_k
        )
        retriever.load_or_create_collection("emphnet_policies")
        
        # Check if collection has data, if not load from inspection file
        collection_count = retriever.collection.count()
        if collection_count == 0:
            st.warning("⚠️ Vector store is empty. Loading chunks from ingestion...")
            chunks_file = "data/chunks_inspection.json"
            if not Path(chunks_file).exists():
                st.error(f"Cannot find {chunks_file}. Please run: python src/ingestion.py")
                return None, None
            chunks = load_chunks_from_json(chunks_file)
            retriever.add_chunks(chunks)
            retriever.persist()
        
        st.success(f"✓ Retriever ready ({collection_count} chunks)")
        
        # Initialize generator
        st.status("🔧 Initializing LLM...", expanded=False)
        generator = OllamaGenerator(
            ollama_host=ollama_host,
            model=ollama_model,
            temperature=0.3,
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
            max_chunks=int(os.getenv("LLM_MAX_CHUNKS", "3")),
            max_chars_per_chunk=int(os.getenv("LLM_MAX_CHARS_PER_CHUNK", "1000")),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "400")),
        )
        st.success(f"✓ LLM ready ({ollama_model})")
        
        return retriever, generator
    
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        st.info("💡 Troubleshooting:")
        st.info("1. Ensure Ollama is running: `ollama serve`")
        st.info("2. Verify PDF ingestion: `python src/ingestion.py`")
        st.info("3. Check .env file configuration")
        return None, None


def format_confidence_badge(confidence: float) -> str:
    """Format confidence score as styled badge"""
    if confidence >= 0.8:
        color = "#4caf50"  # Green
        label = "High"
    elif confidence >= 0.6:
        color = "#ff9800"  # Orange
        label = "Medium"
    else:
        color = "#f44336"  # Red
        label = "Low"
    
    return f'<div class="confidence-badge" style="background-color: {color}; color: white;">Confidence: {label} ({confidence:.0%})</div>'


def main():
    """Main Streamlit application"""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="header">
                <h1>💼 EMPHNET HR Policy Chatbot</h1>
                <p><em>Get instant answers to your HR policy questions</em></p>
            </div>
        """, unsafe_allow_html=True)
    
    # Initialize RAG system
    retriever, generator = initialize_rag_system()
    
    if not retriever or not generator:
        st.stop()
    
    # Language selector
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        language_hint = st.selectbox(
            "💬 Response Language:",
            ["Auto-detect", "English", "العربية"],
            help="The AI will respond in your preferred language"
        )
    
    # Question input
    st.markdown("### 🤔 Ask Your Question")
    
    question = st.text_area(
        "Type your question about HR policies:",
        placeholder="E.g., 'What is the policy on sick leave?' or 'ما هي سياسة الإجازة المرضية؟'",
        height=100,
        label_visibility="collapsed"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")
    
    # Process query
    if submit_button and question.strip():
        with st.spinner("⏳ Searching policy documents..."):
            # Retrieve relevant chunks
            retrieved_chunks = retriever.retrieve(question)
            
            if not retrieved_chunks:
                st.warning("No relevant policy sections found. Please rephrase your question.")
                return
            
            # Generate answer
            with st.spinner("💭 Generating answer..."):
                result = generator.generate(question, retrieved_chunks)
        
        # Display answer
        st.markdown("### ✅ Answer")
        
        answer_col, conf_col = st.columns([3, 1]  )
        with answer_col:
            st.markdown(f"""
                <div class="answer-box">
                    {result['answer']}
                </div>
            """, unsafe_allow_html=True)
        
        with conf_col:
            st.markdown(
                format_confidence_badge(result['confidence']),
                unsafe_allow_html=True
            )
        
        # Display sources
        st.markdown("### 📚 Sources")
        
        with st.expander(f"View {len(result['sources'])} source(s)", expanded=True):
            for i, source in enumerate(result['sources'], 1):
                st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source.get('source', 'Unknown')}<br>
                        <strong>Excerpt:</strong> {source.get('excerpt', 'N/A')}
                    </div>
                """, unsafe_allow_html=True)
        
        # Additional information
        with st.expander("📊 Retrieval Details"):
            st.write(f"**Chunks retrieved:** {len(retrieved_chunks)}")
            st.write(f"**Average relevance score:** {sum(c.get('score', 0) for c in retrieved_chunks) / len(retrieved_chunks):.2%}")
            st.write(f"**Response language:** {result['language']}")
    
    elif submit_button:
        st.warning("Please enter a question before submitting.")
    
    # Footer with help
    st.markdown("---")
    with st.expander("❓ Help & Tips"):
        st.markdown("""
            **How to use this chatbot:**
            1. Type your question in English or Arabic
            2. The AI will search the HR policy manual
            3. Get a grounded answer with source citations
            
            **Example questions:**
            - "What is the sick leave policy?"
            - "How do I request time off?"
            - "What are the maternity leave benefits?"
            - "ما هي سياسة الإجازة الخاصة؟"
            
            **Note:** Answers are strictly based on the official HR policy manual.
            The chatbot will refuse to answer if the information is not found.
        """)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### ℹ️ System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", retriever.collection.count() if retriever.collection else 0)
        with col2:
            st.metric("Model", os.getenv("OLLAMA_MODEL", "Unknown")[:15])
        
        st.markdown("---")
        st.markdown("**Configuration:**")
        st.code(f"""
Host: {os.getenv('OLLAMA_HOST', 'localhost:11434')}
Model: {os.getenv('OLLAMA_MODEL', 'qwen2.5')}
Top-K: {os.getenv('TOP_K_RESULTS', '5')}
        """)
        
        st.markdown("---")
        st.markdown("**Support:**")
        st.info(
            "For issues, ensure Ollama is running and chunks are ingested: "
            "`python src/ingestion.py`"
        )


if __name__ == "__main__":
    main()
