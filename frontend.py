import streamlit as st
from torch import chunk
from backend import chatbot,retrieve_all_history,conn
from langchain_core.messages import HumanMessage,AIMessage
import uuid

def full_reset_app():
    st.session_state.clear()

    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE checkpoint_blobs, checkpoint_writes, checkpoints CASCADE;"
        )

    st.rerun()





def generate_thread_id():
    thread_id= uuid.uuid4()
    return str(thread_id)



def new_chat():
    thread_id=generate_thread_id()
    st.session_state.thread_id=thread_id
    st.session_state.message_history=[]
    add_thread(st.session_state.thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state.all_threads:
        st.session_state.all_threads.append(thread_id)


def load_conversation(thread_id):
    state=chatbot.get_state(config={"configurable":{"thread_id":thread_id}})
    return state.values.get("messages",[])



def rename_threads(thread_id):
    messages=load_conversation(thread_id)
    title=""   
    if not messages:
        title="Start Chatting"
    else:
        title=messages[0].content

    title=title[:40]+"..." if len(title) > 40 else title    
    return title


if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "all_threads" not in st.session_state:
    st.session_state.all_threads = retrieve_all_history()

add_thread(st.session_state.thread_id)


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    new_chat()

if st.sidebar.button("Reset All History"):
    full_reset_app()


st.sidebar.header("Chat history")

for thread_id in st.session_state.all_threads[::-1]:
    if st.sidebar.button(label=str(rename_threads(thread_id)).title(),key=f"thread_btn_{thread_id}"):
        st.session_state.thread_id=thread_id
        messages=load_conversation(thread_id)
        temp_message=[]

        for msg in messages:
            if isinstance(msg,HumanMessage):
                role="user"
            
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                # Fallback for unknown message types
                role = "assistant"            

            temp_message.append({"role":role,"message":msg.content})
        st.session_state.message_history=temp_message


CONFIG={"configurable":{"thread_id":st.session_state.thread_id}}


for message in st.session_state.message_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["message"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["message"])




user_input = st.chat_input("Type your message here...")


if user_input:
    # 1. Show & save user message (same as before)
    st.session_state.message_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Streaming assistant response + collect full text for history
    with st.chat_message("assistant"):
        def stream_response():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        # This displays the stream AND returns the full concatenated string
        full_response = st.write_stream(stream_response())

    # 3. Save the complete assistant message after streaming finishes
    st.session_state.message_history.append(
        {"role": "assistant", "message": full_response}
    )