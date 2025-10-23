# -*- coding: utf-8 -*-
from qdrant_client import QdrantClient, models

def setup_qdrant_collection(collection_name: str, vector_size: int):
    """Initialize an in-memory Qdrant collection."""
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    return qdrant

def upload_documents(qdrant, collection_name, encoder, data):
    """Vectorize and upload documents to Qdrant."""
    qdrant.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=encoder.encode(doc["notes"]).tolist(),
                payload=doc,
            )
            for idx, doc in enumerate(data)
        ]
    )

def search_collection(qdrant, collection_name, encoder, query, limit=3):
    """Search for the most relevant documents."""
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=encoder.encode(query).tolist(),
        limit=limit
    )
    return hits
