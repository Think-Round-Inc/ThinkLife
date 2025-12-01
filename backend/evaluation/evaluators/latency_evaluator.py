"""
Latency Evaluator

Monitors and evaluates response time performance with LangFuse observability
for tracking performance metrics over time.
"""

import logging
import statistics
from typing import Dict, Any, List
from datetime import datetime
from langfuse.decorators import observe, langfuse_context

logger = logging.getLogger(__name__)


class LatencyEvaluator:
    """Evaluates response time performance"""

    def __init__(self):
        """Initialize the latency evaluator"""
        self.response_times: List[float] = []
        
        # Performance thresholds (in seconds)
        self.thresholds = {
            "excellent": 2.0,
            "good": 5.0,
            "acceptable": 10.0,
            "poor": float('inf')
        }

    @observe(name="evaluate_latency")
    async def evaluate(
        self,
        start_time: float,
        end_time: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate response latency
        
        Args:
            start_time: Request start timestamp
            end_time: Response completion timestamp
            context: Additional context for evaluation
            
        Returns:
            Dict containing latency metrics
        """
        latency = end_time - start_time
        self.response_times.append(latency)
        
        # Determine performance category and score
        performance_category, performance_score = self._categorize_latency(latency)
        
        # Calculate running statistics
        stats = self._calculate_statistics()
        
        result = {
            "response_latency_seconds": round(latency, 3),
            "performance_score": performance_score,
            "performance_category": performance_category,
            "statistics": stats,
            "thresholds": self.thresholds,
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluator": "latency"
        }
        
        # Track in LangFuse
        langfuse_context.update_current_trace(
            name="latency_evaluation",
            user_id=context.get("user_id") if context else None,
            session_id=context.get("session_id") if context else None,
            metadata={
                "latency_seconds": latency,
                "performance_category": performance_category,
                "performance_score": performance_score
            }
        )
        
        langfuse_context.update_current_observation(
            output=result,
            metadata={"success": True}
        )
        
        # Log warnings for poor performance
        if performance_category == "poor":
            logger.warning(f"Poor latency detected: {latency:.2f}s (threshold: {self.thresholds['acceptable']}s)")
            langfuse_context.update_current_observation(
                level="WARNING",
                status_message=f"Poor latency: {latency:.2f}s"
            )
        
        return result

    def _categorize_latency(self, latency: float) -> tuple[str, float]:
        """
        Categorize latency and assign performance score
        
        Returns:
            Tuple of (category, score)
        """
        if latency <= self.thresholds["excellent"]:
            return "excellent", 1.0
        elif latency <= self.thresholds["good"]:
            return "good", 0.8
        elif latency <= self.thresholds["acceptable"]:
            return "acceptable", 0.6
        else:
            return "poor", 0.3

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate running statistics for response times"""
        if not self.response_times:
            return {}
        
        # Last 100 responses for recent performance
        recent_times = self.response_times[-100:]
        
        stats = {
            "total_evaluations": len(self.response_times),
            "average_latency_recent": round(statistics.mean(recent_times), 3),
            "median_latency_recent": round(statistics.median(recent_times), 3),
            "min_latency_recent": round(min(recent_times), 3),
            "max_latency_recent": round(max(recent_times), 3),
        }
        
        # Add standard deviation if we have enough data
        if len(recent_times) > 1:
            stats["std_dev_recent"] = round(statistics.stdev(recent_times), 3)
        
        # Add percentiles
        if len(recent_times) >= 10:
            sorted_times = sorted(recent_times)
            stats["p50_latency"] = round(sorted_times[len(sorted_times) // 2], 3)
            stats["p95_latency"] = round(sorted_times[int(len(sorted_times) * 0.95)], 3)
            stats["p99_latency"] = round(sorted_times[int(len(sorted_times) * 0.99)], 3)
        
        return stats

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all recorded latencies
        
        Returns:
            Dict with comprehensive latency statistics
        """
        if not self.response_times:
            return {"error": "No latency data recorded"}
        
        all_stats = self._calculate_statistics()
        
        # Add overall statistics
        all_stats["average_latency_all"] = round(statistics.mean(self.response_times), 3)
        all_stats["median_latency_all"] = round(statistics.median(self.response_times), 3)
        
        # Performance distribution
        categories = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        for latency in self.response_times:
            category, _ = self._categorize_latency(latency)
            categories[category] += 1
        
        all_stats["performance_distribution"] = {
            cat: {
                "count": count,
                "percentage": round(count / len(self.response_times) * 100, 1)
            }
            for cat, count in categories.items()
        }
        
        return all_stats

    def reset_statistics(self):
        """Reset all recorded latency data"""
        self.response_times = []
        logger.info("Latency statistics reset")

