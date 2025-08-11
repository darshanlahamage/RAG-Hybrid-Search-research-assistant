from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from rank_bm25 import BM25Okapi
from config import settings
import pickle
import os

def load_bm25():
    if not os.path.exists("bm25.pkl"):
        raise FileNotFoundError("BM25 index not found.")
    with open("bm25.pkl", "rb") as f:
        bm25, chunks = pickle.load(f)
    return bm25, chunks


def load_chroma():

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=settings.COHERE_API_KEY
    )
    db = Chroma(
        persist_directory=settings.CHROMA_DB_DIR,
        embedding_function= embeddings
    )
    return db

def hybrid_search(query, top_k=5):
    db = load_chroma()
    bm25, chunks = load_bm25()

    semantic_results= db.similarity_search(query, k= top_k)

    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key= lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_results = [chunks[i] for i in bm25_top_indices]

    # merge the bm25 and semantics search results

    all_result = semantic_results + bm25_results

    seen = set()
    unique_results = []
    for r in all_result:
        text = r.page_content
        if text not in seen:
            seen.add(text)
            unique_results.append(r)
    
    return unique_results[:top_k]




if __name__ == "__main__":
    query = "attention mechanism"
    results = hybrid_search(query)
    print(f"\n Results for: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.page_content[:200]}...\n")