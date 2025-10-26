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
        return docs
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

    # Sidebar with project info and dataset details in a fun, engaging style
    st.sidebar.markdown("## 🌱 Project Samarth")
    st.sidebar.info(
        "Hey there, welcome to **Project Samarth!** 🎉\n\n"
        "Powered by official Indian government crop production data, this system answers your questions about "
        "which crops grow where, production volumes, districts & states info — all based on real data from India's "
        "Ministry of Agriculture. 🇮🇳🌾\n\n"
        "**You can ask:**\n"
        "- What's the wheat production in Punjab?\n"
        "- Which districts produce the most rice in Uttar Pradesh?\n"
        "- Total production volume for sugarcane in 2021?\n"
        "- Details about cotton production in Maharashtra.\n\n"
        "Give it a try! Your friendly AI agricultural assistant is here to help. 🚜🤖"
    )

    st.sidebar.header("Sample Questions")
    sample_questions = [
        "What is the wheat production in Punjab?",
        "List top rice-producing districts in Uttar Pradesh.",
        "How much sugarcane was produced in 2021?",
        "Show cotton production details in Maharashtra."
    ]
    for q in sample_questions:
        if st.sidebar.button(q):
            st.session_state['current_question'] = q

    user_question = st.text_area(
        "Enter your question about crop production:",
        value=st.session_state.get('current_question', ""),
        height=100,
        placeholder="Try questions like 'Wheat production in Punjab' or 'Rice production in Uttar Pradesh'"
    )

    if st.button("Get Answer") and user_question:
        with st.spinner("Fetching answer..."):
            qa_chain, vectorstore = initialize_qa_system()
            if qa_chain is None:
                st.error("QA system initialization error. Check logs.")
                return
            result = qa_chain({"query": user_question})
            st.markdown("### Answer")
            st.write(result["result"])
            st.markdown("### Sources")
            sources = format_sources(result["source_documents"])
            for s in sources:
                st.write(f"- {s}")

if __name__ == "__main__":
    main()
