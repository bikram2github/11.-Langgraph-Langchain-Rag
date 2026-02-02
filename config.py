import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


DATA_DIR="data"


INDEX_DIR = "faiss_index"

'''new_document_added = False

pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
if pdf_count > 10:
    new_document_added = True'''


llm=ChatGroq(model="openai/gpt-oss-120b", temperature=0,streaming=True)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



dbname=os.getenv("PG_DB")
db_user=os.getenv("PG_USER")
db_password=os.getenv("PG_PASSWORD")
db_host=os.getenv("PG_HOST")
db_port=os.getenv("PG_PORT")

