from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

file_path1 = "cpc.pdf"
file_path2 = "army_act.pdf"

def create_db(file_path):
    if os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Skipping")
        return
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Created {len(all_splits)} text chunks.")

    print("Creating embeddings using Ollama model...")
    embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(all_splits, embedding=embeddings)

    save_dir = f"faiss_index_{file_path.split('.')[0]}"
    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"✅ FAISS vector database saved successfully to '{save_dir}'.")

for path in [file_path1, file_path2]:
    create_db(path)