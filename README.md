# 🌾 Project Samarth - Agricultural Q&A System

An intelligent Q&A system for analyzing India's agricultural commodity price data from data.gov.in using RAG (Retrieval Augmented Generation).

## 🎯 Features

- **5,573+ Government Datasets** indexed from data.gov.in
- **Llama 3.3 70B** LLM via Groq API for intelligent responses
- **ChromaDB** vector database for semantic search
- **Real-time commodity prices** from Ministry of Agriculture
- **Source citations** for every answer

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Llama 3.3 (via Groq API)
- **Vector DB:** ChromaDB
- **Data Source:** data.gov.in API
- **Embeddings:** HuggingFace all-MiniLM-L6-v2

## 📊 Dataset

District-wise, Season-wise Crop Production Statistics
- Source: Open Government Data (OGD) Platform India
- URL: https://data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0

## 🚀 Local Setup

Clone repository
git clone https://github.com/YOUR_USERNAME/project-samarth.git
cd project-samarth

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Set up environment variables
Create .env file with:
GROQ_API_KEY=your_groq_api_key
DATAGOVINDIA_API_KEY=your_data_gov_key
Run data pipeline (optional - pre-processed data included)
python scripts/01_fetch_data.py
python scripts/02_process_data.py
python scripts/03_create_vectordb.py

Launch app
streamlit run app.py
## 💬 Sample Questions

- "What is the wheat price in Jaunpur market?"
- "Show me all commodity prices in Uttar Pradesh"
- "Compare wheat prices across different districts"
- "What is the price of tomato in Punjab?"

## 📝 Project Structure
project-samarth/
├── app.py # Main Streamlit application
├── config.py # Configuration settings
├── requirements.txt # Python dependencies
├── .env # Environment variables (not in git)
├── data/
│ ├── raw/ # Raw data from API
│ └── processed/ # Processed data
├── chroma_db/ # Vector database
└── scripts/
├── 01_fetch_data.py
├── 02_process_data.py
└── 03_create_vectordb.py


## 🌐 Live Demo

[Link to deployed app]

