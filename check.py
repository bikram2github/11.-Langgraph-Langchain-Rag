import os

from config import DATA_DIR


pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])

print(f"Number of PDF documents in '{DATA_DIR}': {pdf_count}")



