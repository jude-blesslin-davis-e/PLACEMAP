from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_fireworks import Fireworks
import logging

# ===== Logger Setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FastAPI App Setup =====
app = FastAPI()

# ===== Enable CORS for Kotlin App =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this later to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Set Fireworks API Key =====
fireworks_api_key = "fw_3ZJ3B2SNxCm8Tr9JnWUavrks"
os.environ["FIREWORKS_API_KEY"] = fireworks_api_key

# ===== Load CSV Data =====
csv_path = r"C:\Users\admin\OneDrive\Desktop\Pd lab repo.csv"
vc = None

try:
    if os.path.exists(csv_path):
        logger.info(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)

        if df.empty:
            logger.error("CSV file is empty.")
            raise ValueError("CSV file is empty.")

        docs = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vc = FAISS.from_documents(split_docs, embedding_model)

        logger.info("Vector store initialized successfully.")
    else:
        logger.error(f"CSV file not found at: {csv_path}")
except Exception as e:
    logger.error(f"Error loading CSV: {e}")

# ===== Initialize LLM =====
llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-405b-instruct",
    base_url="https://api.fireworks.ai/inference/v1/completions",
    temperature=0.3
)

# ===== Request Body Model =====
class ChatbotRequest(BaseModel):
    dream_company: str
    domain: str
    skills: list[str]
    skill_levels: list[int]

# ===== Health Check Endpoint =====
@app.get("/")
async def root():
    return {"message": "Server is running"}

# ===== Helper Functions =====
def retrieve_context(query: str) -> str:
    """Retrieve relevant information from FAISS vector store."""
    if vc is None:
        logger.error("Vector store is not initialized.")
        return "Vector store is not initialized."

    try:
        retrieved_docs = vc.similarity_search(query, k=5)
        return "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant data found."
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return "Error retrieving context."

def generate_response(dream_company: str, domain: str, skills: list[str]) -> dict:
    """Generate chatbot response including company overview and insights."""
    query = f"{dream_company} {domain}"
    retrieved_context = retrieve_context(query)

    template = f"""
    Dream Company: {dream_company}
    Domain: {domain}
    Skills Provided: {', '.join(skills)}

    Company Overview:
    Provide a brief overview of {dream_company}. Include details such as industry, major services/products, and notable achievements.

    Insights:
    - Role of {dream_company} in the {domain} domain.
    - Required Skills for working in {domain} at {dream_company}.
    - Recommendations: Courses, certifications, or projects to improve suitability.
    """

    try:
        logger.info("Sending request to Fireworks API...")
        response = llm.invoke(template, max_tokens=1000)
        logger.info(f"Fireworks Response: {response}")

        return {
            "response": response
        }
    except Exception as e:
        logger.error(f"Error generating response from Fireworks API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")

# ===== Chatbot API Endpoint =====
@app.post("/chatbot")
async def chatbot(request: ChatbotRequest):
    """Handles chatbot request and generates response."""
    if vc is None:
        logger.error("Vector store is not initialized.")
        raise HTTPException(status_code=500, detail="Vector store is not initialized due to CSV loading error.")

    try:
        logger.info(f"Received chatbot request: {request.dict()}")
        result = generate_response(
            request.dream_company,
            request.domain,
            request.skills
        )
        logger.info(f"Chatbot response generated: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing chatbot request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
