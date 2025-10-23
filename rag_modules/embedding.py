# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer

def get_encoder(model_name: str = "all-MiniLM-L6-v2"):
    """Load SentenceTransformer encoder."""
    encoder = SentenceTransformer(model_name)
    return encoder

def create_embeddings(encoder, texts):
    """Generate embeddings for a list of texts."""
    return [encoder.encode(text).tolist() for text in texts]
