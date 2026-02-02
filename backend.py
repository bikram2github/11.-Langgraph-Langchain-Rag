import os
from config import dbname, db_user, db_host,db_password,db_port, llm, embeddings, INDEX_DIR, DATA_DIR
from chatbot import load_document, text_split, generate_embeddings, generate_retriever,generate_rag_chain

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage


from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

#import sqlite3
#from langgraph.checkpoint.sqlite import SqliteSaver
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver





#build rag

docs=load_document(DATA_DIR)

docs=text_split(docs)

vectorstore=generate_embeddings(docs,DATA_DIR,INDEX_DIR,embeddings)

retriever=generate_retriever(vectorstore)

rag_chain=generate_rag_chain(retriever,llm)




#Langgraph PArt

#State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def ask(state: ChatState):
    question = state["messages"][-1].content
    chat_history = state["messages"]

    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return {
        "messages": [AIMessage(content=response)]
    }


"""def chat_node(state:ChatState):
    message=state["messages"]
    response=llm.invoke(message)

    return {"messages": [response]}"""

#conn=sqlite3.connect(database="chatbot1.db",check_same_thread=False)


conn = psycopg.connect(
    dbname=dbname,
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port,
    autocommit=True
)



checkpoint_saver=PostgresSaver(conn)
checkpoint_saver.setup()

graph=StateGraph(ChatState)

graph.add_node("rag",ask)

graph.add_edge(START,"rag")
graph.add_edge("rag",END)

chatbot=graph.compile(checkpointer=checkpoint_saver)

def retrieve_all_history():
    all_threads=set()
    for checkpoint in checkpoint_saver.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)


'''response=chatbot.invoke({"messages":[HumanMessage(content="Hello")]},config={"configurable":{"thread_id":"test_once"}})

print(response)'''