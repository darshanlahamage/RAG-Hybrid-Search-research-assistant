import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from config import settings
from langchain_community.document_transformers import EmbeddingsRedundantFilter
import re
from rank_bm25 import BM25Okapi
import pickle

def clean_text(text: str) -> str:
    # Remove multiple spaces, citations like [12], and weird unicode
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_pdfs(data_folder: str):

    documents =[]
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, file))
            docs =loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata.update({"source": file})
            documents.extend(docs)
    
    return documents

def chunk_documents(documents):
    semantic_splitter = SemanticChunker(
        CohereEmbeddings(
            model = settings.EMBEDDING_MODEL,
            cohere_api_key= settings.COHERE_API_KEY
        ),
        breakpoint_threshold_type="percentile"
    )

    chunk_size: int = 800
    chunk_overlap: int = 160

    semantically_chunked = semantic_splitter.split_documents(documents)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap
    )
    final_chunks = recursive_splitter.split_documents(semantically_chunked)
    return final_chunks

def store_in_chroma(chunks):

    embeddings = CohereEmbeddings(
    model=settings.EMBEDDING_MODEL,
    cohere_api_key=settings.COHERE_API_KEY
    )

    db = Chroma(
        persist_directory=settings.CHROMA_DB_DIR,
        embedding_function= embeddings
    )

    db.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in Chroma at {settings.CHROMA_DB_DIR}")

def create_bm25_index(chunks):
    tokenized_corpus = [doc.page_content.split(" ") for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open("bm25.pkl", "wb") as f:
        pickle.dump((bm25, chunks), f)
    print("BM25 index saved as bm25.pkl")

if __name__== "__main__":
    docs = load_pdfs('data')
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    store_in_chroma(chunks)
    create_bm25_index(chunks)



