import os
import fitz
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or input("Enter Google API key: ")
os.environ["GOOGLE_API_KEY"] = api_key

documents, qa_chains = {}, {}

def extract_text(pdf_path):
    with fitz.open(pdf_path) as pdf:
        return "\n".join([f"--- PAGE {i+1} ---\n{p.get_text()}" for i, p in enumerate(pdf)])

def split_text(text, name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return [Document(page_content=c, metadata={"source": name, "chunk": i})
            for i, c in enumerate(splitter.split_text(text))]

def create_qa_chain(docs, name):
    db_path = f"db_{name.replace(' ', '_')}"
    shutil.rmtree(db_path, ignore_errors=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    prompt = PromptTemplate(
        template=f"""
You are a helpful assistant for the document "{name}".
Use the context below to answer questions. Be friendly for casual greetings.

Context: {{context}}
Question: {{question}}
Answer:
""",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def add_pdf(pdf_path, doc_name=None):
    name = doc_name or os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Adding: {name}")
    docs = split_text(extract_text(pdf_path), name)
    qa_chains[name] = create_qa_chain(docs, name)
    documents[name] = {"path": pdf_path, "chunks": len(docs)}
    print(f"✓ {name} ready! ({len(docs)} chunks)")

def load_pdfs():
    print("=== Add Your PDFs ===")
    while True:
        path = input("\nPDF path (or 'done'): ").strip()
        if path.lower() == "done": break
        if not os.path.exists(path):
            print("File not found!"); continue
        add_pdf(path, input("Document name (or Enter): ").strip() or None)
    print(f"\nLoaded {len(documents)} documents: {list(documents.keys())}")

def chat():
    if not documents:
        print("No documents loaded!"); return
    current = next(iter(documents))
    print(f"\nChatting with: {current}")
    print("Commands: list | use <name> | add | exit")

    while True:
        q = input(f"\n[{current}] You: ").strip()
        if q.lower() == "exit": break
        if q.lower() == "list": print(list(documents.keys())); continue
        if q.lower().startswith("use "):
            name = q[4:]; 
            current = name if name in documents else current
            print(f"Switched to: {current}"); continue
        if q.lower() == "add":
            path = input("PDF path: ")
            if os.path.exists(path):
                add_pdf(path, input("Document name (or Enter): ").strip() or None)
                current = list(documents.keys())[-1]
            else: print("File not found!")
            continue
        if q: print(f"\nAI: {qa_chains[current].invoke({'query': q})['result']}")

def main():
    print("🤖 Multi-PDF Chat Bot")
    load_pdfs()
    chat()

if __name__ == "__main__":
    main()
