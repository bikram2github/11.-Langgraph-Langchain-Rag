'''from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
'''

#import shutil
#from config import new_document_added

from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from dotenv import load_dotenv


import os

load_dotenv()



def load_document(path):
    loader = DirectoryLoader(
    path=path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
    )

    docs = loader.load()

    return docs

def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(docs)
    return docs


def generate_embeddings(docs,DATA_DIR,INDEX_DIR,embeddings):
            
    base_name = os.path.splitext(DATA_DIR)[0]
    base_name = base_name.replace(" ", "_").replace(".", "_").replace("/", "_")
    INDEX_PATH = os.path.join(INDEX_DIR, f"{base_name}_index")

    '''if new_document_added:
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
'''
    os.makedirs(INDEX_DIR, exist_ok=True)

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)

    return vectorstore


def generate_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return retriever


def generate_rag_chain(retriever, llm):

    system_prompt = (
        "You are a helpful and precise AI assistant. "
        "Answer the user's question strictly based on the provided context. "        
        "if anyone ask you 'who are you', respond with 'I am an AI language model created by BIKRAM MAITY to assist with your questions about Machine Learning.' "
        "if anyone ask you how are you, respond with 'Hello! I am Good, How can I assist you today?' "
        "if anyone ask you hii or hello, respond with 'Hello! How can I assist you today?' "
        "If the context does not contain enough information to answer, respond with: 'I don't know.' "
        "strictly Do not use any external or prior knowledge. "
        "strictly If the user makes grammar mistakes, fix them in your answer. "
        "Keep your answer clear, concise, and directly relevant to the context below.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["input"]) | retriever,
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | prompt| llm | StrOutputParser()
    )
    return rag_chain

