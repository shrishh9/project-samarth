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
        prompt_template = """You are an assistant for agricultural market prices and arrivals in India.
You have access to daily arrivals data that includes:
- state, district, market
- commodity (crop)
- variety, grade
- arrival date
- minimum, maximum, and modal prices

Always answer with specific prices, varieties, markets, or districts as per the question.
Cite your source as 'Agmarknet (Government of India) daily market arrivals and prices'.

Context: {context}

Question: {question}

Your answer:
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
        st.error("Error loading agricultural market data.")
        return None

def format_sources(source_documents):
    return ["Agmarknet (Government of India) daily market arrivals and prices"]

def main():
    st.markdown('<h1 class="main-header">🌾 Project Samarth</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">India Market Price & Crop Arrivals Q&A System</p>', unsafe_allow_html=True)

    st.sidebar.markdown("## 🏪 About This Project")
    st.sidebar.info(
        "**Project Samarth is your AI-powered mandi insights buddy!**\n\n"
        "📉 Uses real, daily market arrivals and price data from Agmarknet.\n\n"
       
        "**All answers cite 'Agmarknet (Government of India) daily market arrivals and prices'.**"
    )

    st.sidebar.header("Sample Questions")
    sample_questions = [
        "What is the modal price of tomato in Chittoor on 25/10/2025?",
        "What is the highest price for banana in Mehsana on 25/10/2025?",
        "List commodities traded in Guntur market on 25/10/2025.",
        "What varieties of cabbage were traded in Amreli?",
        "Price range for dry chillies in Guntur on 25/10/2025?"
    ]
    for q in sample_questions:
        if st.sidebar.button(q):
            st.session_state['current_question'] = q

    user_question = st.text_area(
        "Ask about mandi prices, commodities, districts, varieties, dates:",
        value=st.session_state.get('current_question', ""),
        height=100,
        placeholder="E.g. 'Modal price of paddy in Krishna on 25/10/2025'"
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
            st.markdown("**Source:** Agmarknet (Government of India) daily market arrivals and prices")

if __name__ == "__main__":
    main()
