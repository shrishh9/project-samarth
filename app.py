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
import json

# Increase recursion limit
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
def load_documents():
    try:
        with open("data/processed/documents.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        df = pd.DataFrame([d["metadata"] for d in docs])
        return df
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        st.error("Error loading crop production data.")
        return None

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
    df_docs = load_documents()

    if qa_chain is None or df_docs is None:
        st.error("System initialization failed, check logs.")
        return

    st.sidebar.header("Sample Questions")
    sample_questions = [
        "What is the wheat production in Punjab?",
        "Which state produced the most rice in 2020?",
        "List the wheat production by district in Haryana.",
        "What is the total crop production in Uttar Pradesh?",
    ]
    for q in sample_questions:
        if st.sidebar.button(q):
            st.session_state['current_question'] = q

    user_question = st.text_area("Enter your question about agriculture data:",
                                 value=st.session_state.get('current_question', ""),
                                 height=100,
                                 placeholder="Try questions like 'Wheat production in Punjab'")

    if st.button("Get Answer") and user_question:
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
