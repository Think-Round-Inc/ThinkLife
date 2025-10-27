import statistics
from datetime import datetime
from typing import Dict, Any

class LatencyEvaluator:
    def __init__(self):
        self.response_times = []

    async def evaluate(
        self,
        start_time: float,
        end_time: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        latency = end_time - start_time
        self.response_times.append(latency)
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