import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import *
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Project Samarth - Agricultural Q&A System",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_qa_system():
    """
    Initialize the QA system with ChromaDB and Groq
    """
    try:
        logger.info("Initializing QA system...")
        
        # Initialize ChromaDB client
        logger.info(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get collection
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Loaded collection with {collection.count()} documents")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create LangChain Chroma vectorstore
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        
        # Initialize Groq LLM
        logger.info(f"Initializing Groq with Llama 3")
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=2,
        )
        
        # Create prompt template
        prompt_template = """You are an agricultural data analyst assistant for the Indian Government's data portal.

Use the following context from official government datasets to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the available data to answer this question."

Always cite the source dataset in your answer.

Context: {context}

Question: {question}

Answer (provide a clear, concise response with data citations):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA system initialized successfully!")
        return qa_chain, vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing QA system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None, None

def format_sources(source_documents):
    """Format source documents for display"""
    sources = []
    seen_sources = set()
    
    for doc in source_documents:
        metadata = doc.metadata
        source_key = f"{metadata.get('source', 'Unknown')}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                'Source': metadata.get('source', 'Unknown'),
                'URL': metadata.get('dataset_url', 'N/A')
            })
    
    return sources

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Project Samarth</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Q&A System for Indian Agricultural Data</p>', unsafe_allow_html=True)
    
    # Initialize QA system
    qa_chain, vectorstore = initialize_qa_system()
    
    if qa_chain is None:
        st.error("❌ Failed to initialize the QA system. Please check the logs.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.write("""
        This system answers questions about India's agricultural economy
        using data from **data.gov.in**.
        
        **Powered by:**
        - 🦙 Llama 3 (via Groq API)
        - 🗂️ ChromaDB Vector Database
        - 📊 5,573+ Government Datasets
        """)
        
        st.header("📝 Sample Questions")
        sample_questions = [
            "What is the wheat price in Uttar Pradesh today?",
            "Show me tomato prices across different districts",
            "Compare ginger prices between markets",
            "What commodities are available in Jaunpur market?",
            "List all crops with their prices in Punjab"
        ]
        for q in sample_questions:
            if st.button(q, key=q):
                st.session_state['current_question'] = q
    
    # Main content
    st.header("💬 Ask Your Question")
    
    default_question = st.session_state.get('current_question', '')
    user_question = st.text_area(
        "Enter your question about agriculture data:",
        value=default_question,
        height=100,
        placeholder="e.g., What is the wheat price in Uttar Pradesh today?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Get Answer", type="primary")
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state['current_question'] = ''
            st.rerun()
    
    # Process question
    if ask_button and user_question:
        with st.spinner("🔍 Analyzing data from government sources..."):
            try:
                result = qa_chain({"query": user_question})
                
                st.markdown("---")
                st.header("📋 Answer")
                st.write(result['result'])
                
                st.markdown("---")
                st.header("📚 Data Sources")
                
                sources = format_sources(result['source_documents'])
                
                if sources:
                    for i, source in enumerate(sources[:5], 1):
                        with st.expander(f"Source {i}: {source['Source']}"):
                            st.write(f"**URL:** {source['URL']}")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
                logger.error(f"Error: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📊 Data Source: Open Government Data (OGD) Platform India</p>
        <p>🚀 Powered by Llama 3 (Groq), ChromaDB & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
