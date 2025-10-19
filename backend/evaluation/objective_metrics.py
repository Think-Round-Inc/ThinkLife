import statistics
from datetime import datetime
from typing import Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

class ObjectiveMetricsEvaluator:
    """Handles objective, computable metrics: relevance (embeddings) and latency (timing)."""

    def __init__(self, hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.hf_embeddings = HuggingFaceEmbeddings(model_name=hf_model)
        
        self.response_times = []

    async def evaluate_relevance(self, user_query: str, bot_response: str):
        """
        Evaluate relevance of bot response to user query using cosine similarity.
        Returns a relevance score between 0.0 and 1.0.
        """
        # Embed both user query and bot response
        user_emb = self.hf_embeddings.embed_query(user_query)
        bot_emb = self.hf_embeddings.embed_query(bot_response)

        # Compute cosine similarity using sklearn
        relevance_score = cosine_similarity([user_emb], [bot_emb])[0][0]


        relevance_score = max(0.0, min(1.0, relevance_score))

        return {
            "relevance_score": relevance_score
        }

    async def evaluate_latency(
        self,
        start_time: float,
        end_time: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate response latency.

        Args:
            start_time: Request start timestamp
            end_time: Response completion timestamp
            context: Additional context for evaluation

        Returns:
            Dict containing latency metrics
        """
        latency = end_time - start_time
        self.response_times.append(latency)

        # Performance thresholds (in seconds)
        excellent_threshold = 2.0
        good_threshold = 5.0
        acceptable_threshold = 10.0

        if latency <= excellent_threshold:
            performance_score = 1.0
            performance_category = "excellent"
        elif latency <= good_threshold:
            performance_score = 0.8
            performance_category = "good"
        elif latency <= acceptable_threshold:
            performance_score = 0.6
            performance_category = "acceptable"
        else:
            performance_score = 0.3
            performance_category = "poor"

        
        avg_latency = statistics.mean(self.response_times[-100:])  

        result = {
            "response_latency": latency,
            "performance_score": performance_score,
            "performance_category": performance_category,
            "average_latency_recent": avg_latency,
            "threshold_excellent": excellent_threshold,
            "threshold_good": good_threshold,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        return result