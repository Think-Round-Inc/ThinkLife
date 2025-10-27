import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

class RelevanceEvaluator:
    def __init__(self, hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.hf_embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    async def evaluate(self, user_query: str, bot_response: str):
        user_emb = self.hf_embeddings.embed_query(user_query)
        bot_emb = self.hf_embeddings.embed_query(bot_response)
        relevance_score = cosine_similarity([user_emb], [bot_emb])[0][0]
        relevance_score = max(0.0, min(1.0, relevance_score))
        return {
            "relevance_score": relevance_score
        }