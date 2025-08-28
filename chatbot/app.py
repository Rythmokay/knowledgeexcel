import streamlit as st
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class UserInfo(BaseModel):
    name: str = "NULL"
    college: str = "NULL"
    summary: str

st.set_page_config(page_title="Simple Chatbot", layout="centered")
st.title("🤖 Simple Chatbot with Schema Classes")

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1)

def get_response(user_input):
    history = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history += f"User: {msg['content']}\n"
        else:
            history += f"AI: {msg['content'].summary}\n"

    prompt = f"""You must respond ONLY in this exact JSON format:
{{
    "name": "NULL",
    "college": "NULL",
    "summary": "what the user has said or done"
}}

Extract name and college from all messages. Write summary describing what user actually said.

Conversation so far: {history}
Current user message: {user_input}

JSON only:"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1]
        
        data = json.loads(content.strip())
        return UserInfo(**data)
        
    except Exception as e:
        return UserInfo(
            name="NULL",
            college="NULL", 
            summary=f"User said: {user_input}"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            user_info = msg["content"]
            st.write(f"**Name:** {user_info.name}")
            st.write(f"**College:** {user_info.college}")
            st.write(f"**Summary:** {user_info.summary}")

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })
    
    ai_response = get_response(user_input)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_response
    })
    
    st.rerun()

with st.sidebar:
    st.header("Controls")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.messages:
        st.header("Latest Info")
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                latest_info = msg["content"]
                st.write(f"Name: {latest_info.name}")
                st.write(f"College: {latest_info.college}")
                break
