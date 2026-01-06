from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

file_path1 = "cpc.pdf"
file_path2 = "army_act.pdf"

def create_db(file_path):
    save_dir = f"faiss_index_{file_path.split('.')[0]}"
    
    if os.path.exists(save_dir):
        print(f"Index '{save_dir}' already exists. Skipping.")
        return
        
    if not os.path.exists(file_path):
        print(f"Input file '{file_path}' not found. Skipping.")
        return

    print(f"Loading PDF with Unstructured: {file_path}")
    loader = UnstructuredPDFLoader(file_path)
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

    print("Creating embeddings using HuggingFace model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'} 
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(all_splits, embedding=embeddings)

    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"✅ FAISS vector database saved successfully to '{save_dir}'.")

for path in [file_path1, file_path2]:
    create_db(path)

