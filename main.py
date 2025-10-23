# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from rag_modules.data_ingestion import load_and_prepare_data
from rag_modules.embedding import get_encoder
from rag_modules.retrieval import setup_qdrant_collection, upload_documents, search_collection
from rag_modules.llm import query_llm

# Load environment variables from .env
load_dotenv()

CSV_PATH = "data/top_rated_wines.csv"
COLLECTION_NAME = "top_wines"

# Get API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("APIkey not found! Please set OPENAI_API_KEY in .env file.")

USER_PROMPT = "Suggest me an amazing Malbec from Argentina."

def main():
    # 1. Load data
    data = load_and_prepare_data(CSV_PATH)
    print(f"Loaded {len(data)} records.")

    # 2. Initialize encoder
    encoder = get_encoder()

    # 3. Create Qdrant collection
    qdrant = setup_qdrant_collection(COLLECTION_NAME, encoder.get_sentence_embedding_dimension())

    # 4. Upload documents
    upload_documents(qdrant, COLLECTION_NAME, encoder, data)

    # 5. Search
    hits = search_collection(qdrant, COLLECTION_NAME, encoder, USER_PROMPT)
    search_results = [hit.payload for hit in hits]
    print("\nTop Search Results:")
    for hit in hits:
        print(hit.payload["title"] if "title" in hit.payload else hit.payload, "→ score:", hit.score)

    # 6. Query LLM
    response = query_llm(API_KEY, USER_PROMPT, search_results)
    print("\nLLM Response:\n", response)

if __name__ == "__main__":
    main()
