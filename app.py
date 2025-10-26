import sys
import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import *
import logging
import pandas as pd
from utils.structured_queries import query_crop_production, query_rainfall_data
from utils.analysis import analyze_trend
from utils.synthesis import synthesize_policy_recommendations

# Increase recursion limit (temporary workaround)
sys.setrecursionlimit(3000)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Project Samarth", page_icon="🌾", layout="wide")

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

@st.cache_resource(ttl=3600)
def initialize_qa_system():
    try:
        logger.info("Initializing QA system...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(client=chroma_client, collection_name=COLLECTION_NAME, embedding_function=embeddings)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
        
        # Simplified prompt to prevent recursion error
        prompt_template = """Answer the question briefly based on the context:
        
Context: {context}

Question: {question}
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("QA system initialized successfully!")
        return qa_chain, vectorstore
    except Exception as e:
        logger.error(f"Error initializing QA system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None, None

@st.cache_resource(ttl=1800)
def load_datasets():
    try:
        crops_df = pd.read_json("data/processed/documents.json")
        rainfall_df = pd.read_csv("data/processed/rainfall.csv")  # Adjust path accordingly if needed
        return crops_df, rainfall_df
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        st.error("Error loading datasets.")
        return None, None

def handle_complex_question(question, crops_df, rainfall_df):
    # Sample fixed parameters for demo
    state_x = "Punjab"
    state_y = "Haryana"
    years = list(range(2015, 2024))
    crop_type = "Wheat"
    top_m = 5

    rainfall_x = query_rainfall_data(rainfall_df, state_x, years)
    rainfall_y = query_rainfall_data(rainfall_df, state_y, years)
    crop_x = query_crop_production(crops_df, state_x, crop_type, years)
    crop_y = query_crop_production(crops_df, state_y, crop_type, years)

    trend_x = analyze_trend(rainfall_x, crop_x)
    trend_y = analyze_trend(rainfall_y, crop_y)

    synthesized_answer = synthesize_policy_recommendations(trend_x, trend_y, crop_x, crop_y)
    return synthesized_answer

def format_sources(source_documents):
    sources = []
    seen = set()
    for doc in source_documents:
        src = doc.metadata.get('source', 'Unknown')
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources

def main():
    st.markdown('<h1 class="main-header">🌾 Project Samarth</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Q&A System for Indian Agricultural Data</p>', unsafe_allow_html=True)

    qa_chain, vectorstore = initialize_qa_system()
    crops_df, rainfall_df = load_datasets()

    if qa_chain is None or crops_df is None or rainfall_df is None:
        st.error("System initialization failed, check logs.")
        return

    st.sidebar.header("Sample Questions")
    sample_questions = [
        "What is the wheat price in Jaunpur market?",
        "Show me all commodity prices in Uttar Pradesh",
        "Compare wheat prices between districts",
        "What is the rainfall and production trend for rice in State_X and State_Y in last decade?"
    ]
    for q in sample_questions:
        if st.sidebar.button(q):
            st.session_state['current_question'] = q

    user_question = st.text_area("Enter your question about agriculture data:",
                                 value=st.session_state.get('current_question', ""),
                                 height=100,
                                 placeholder="Try questions like 'Wheat prices in Uttar Pradesh'")

    if st.button("Get Answer") and user_question:
        if any(keyword in user_question.lower() for keyword in ["compare", "trend", "correlate", "policy"]):
            with st.spinner("Processing complex query with multi-source data..."):
                answer = handle_complex_question(user_question, crops_df, rainfall_df)
                st.markdown("### Synthesized Answer")
                st.write(answer)
        else:
            with st.spinner("Fetching answer..."):
                result = qa_chain({"query": user_question})
                st.markdown("### Answer")
                st.write(result["result"])
                st.markdown("### Sources")
                sources = format_sources(result["source_documents"])
                for s in sources:
                    st.write(f"- {s}")

if __name__ == "__main__":
    main()
