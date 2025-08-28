import os
import fitz
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #EAEAEA;
    }
    .user-bubble {
        background-color: #4CAF50;
        color: white;
        border-radius: 15px 15px 0 15px;
        padding: 12px 18px;
        margin: 8px 30% 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        max-width: 60%;
        word-wrap: break-word;
    }
    .bot-bubble {
        background-color: #1E1E1E;
        color: #EAEAEA;
        border-radius: 15px 15px 15px 0;
        padding: 12px 18px;
        margin: 8px 0 8px 30%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        max-width: 60%;
        word-wrap: break-word;
    }
    .chat-container {
        background-color: #1A1A1A;
        border-radius: 10px;
        padding: 15px;
        height: 500px;
        overflow-y: auto;
        border: 1px solid #333;
        margin-bottom: 15px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #555;
        background-color: #2A2A2A;
        color: white;
    }
    .send-btn {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 6px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

class QuestionInput(BaseModel):
    question: str = Field(..., min_length=0, max_length=5000, description="Question about the PDF or general.")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

llm = GoogleGenerativeAI(model="gemini-1.5-flash")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="sidebar_uploader")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        doc = fitz.open("temp.pdf")
        full_text = "\n".join([page.get_text() for page in doc])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([full_text])
        st.session_state.num_chunks = len(docs)
        print("These are my docs" ,docs)

        vectorstore = FAISS.from_documents(docs, embedding)
        retriever = vectorstore.as_retriever()

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
        )

        st.success("✅ PDF processed successfully!")
        print("session state answer - " , st.session_state.qa_chain)
        
        st.markdown(f"**Chunks stored in memory:** {st.session_state.num_chunks}")
    else:
        st.info("Please upload a PDF document to start or ask general questions.")

def render_chat_message(message, is_user=True):
    bubble_class = "user-bubble" if is_user else "bot-bubble"
    st.markdown(f"<div class='{bubble_class}'>{message}</div>", unsafe_allow_html=True)

st.title("📄 RAG Chatbot (Gemini + PyMuPDF)")

chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.info("Ask a question to start chatting!")
    else:
        st.markdown("", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            render_chat_message(chat["question"], is_user=True)
            render_chat_message(chat["answer"], is_user=False)
        st.markdown("", unsafe_allow_html=True)

with st.form("question_form", clear_on_submit=True):
    user_question = st.text_input("💬 Ask a question about the document or general")
    submitted = st.form_submit_button("Send")

    if submitted:
        user_question_stripped = user_question.strip()
        if len(user_question_stripped) < 1:
            st.error("Please enter a question with at least 1 characters.")
        else:
            try:
                validated = QuestionInput(question=user_question_stripped)

                if st.session_state.qa_chain:
                    answer = st.session_state.qa_chain.run(validated.question)
                else:
                    answer = llm(validated.question)

                st.session_state.chat_history.append({
                    "question": validated.question,
                    "answer": answer,
                })

                st.rerun()

            except ValidationError as e:
                st.error(f"❌ Validation error: {e}")
