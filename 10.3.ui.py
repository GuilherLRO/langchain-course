import streamlit as st
import importlib.util
import sys
from dotenv import load_dotenv
load_dotenv()

# Import the core module (handles filename with dots)
spec = importlib.util.spec_from_file_location("core_module", "10.2.core.py")
core_module = importlib.util.module_from_spec(spec)
sys.modules["core_module"] = core_module
spec.loader.exec_module(core_module)
run_llm_chain = core_module.run_llm_chain

# Page configuration
st.set_page_config(
    page_title="LangChain QA Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #262730;
    }
    .source-link {
        color: #1f77b4;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 LangChain QA Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        Ask questions about LangChain documentation. The assistant will retrieve relevant information and provide answers.
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This QA assistant uses:
    - **RAG (Retrieval-Augmented Generation)**
    - **Pinecone Vector Store**
    - **OpenAI GPT-4o-mini**
    
    It retrieves relevant documents from the LangChain documentation and generates answers based on the retrieved context.
    """)
    
    st.header("📊 Statistics")
    st.metric("Questions Asked", len(st.session_state.history))
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    query = st.text_input(
        "Ask a question:",
        placeholder="e.g., What is langchain? How do I create a chain?",
        key="query_input"
    )

with col2:
    st.write("")  # Spacing
    submit_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Process query
if submit_button and query:
    with st.spinner("🔍 Searching and generating answer..."):
        try:
            # Run the chain
            result = run_llm_chain(query)
            
            # Store in history
            st.session_state.history.append({
                "query": query,
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_docs"]
            })
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Display source links
            st.markdown("### 📄 Sources")
            sources = [doc.metadata.get('source', 'unknown') for doc in result["retrieved_docs"] if doc.metadata.get('source', 'unknown') != 'unknown']
            if sources:
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
            else:
                st.markdown("No sources available")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Question History")
    
    # Display history in reverse order (newest first)
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Q: {item['query']}", expanded=False):
            st.markdown("**Answer:**")
            st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
            st.markdown("**Sources:**")
            sources = [doc.metadata.get('source', 'unknown') for doc in item['retrieved_docs'] if doc.metadata.get('source', 'unknown') != 'unknown']
            if sources:
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
            else:
                st.markdown("No sources available")

