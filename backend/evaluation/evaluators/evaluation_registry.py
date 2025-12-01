"""
Evaluation Registry

Coordinates all evaluators and provides unified interface with LangFuse observability
for comprehensive evaluation, monitoring, and tracing of AI responses.
"""

import asyncio
import logging
import statistics
from typing import Dict, Any, Optional
from datetime import datetime
from langfuse.decorators import observe, langfuse_context

from .empathy_evaluator import EmpathyEvaluator
from .trigger_evaluator import TriggerEvaluator
from .crisis_evaluator import CrisisEvaluator
from .latency_evaluator import LatencyEvaluator
from .accessibility_evaluator import AccessibilityEvaluator

logger = logging.getLogger(__name__)


class EvaluationRegistry:
    """
    Registry that manages and coordinates all evaluators with LangFuse observability
    
    Provides:
    - Unified evaluation interface
    - Concurrent evaluation execution
    - Aggregate scoring
    - LangFuse tracing and monitoring
    - Performance metrics
    """

    def __init__(self, llm_client=None):
        """
        Initialize evaluation registry
        
        Args:
            llm_client: LLM client for evaluators that need it
        """
        self.llm_client = llm_client
        
        # Initialize evaluators
        self.empathy_evaluator = EmpathyEvaluator(llm_client=llm_client)
        self.trigger_evaluator = TriggerEvaluator(llm_client=llm_client)
        self.crisis_evaluator = CrisisEvaluator(llm_client=llm_client)
        self.latency_evaluator = LatencyEvaluator()
        self.accessibility_evaluator = AccessibilityEvaluator(llm_client=llm_client)
        
        logger.info("Evaluation Registry initialized with all evaluators")

    @observe(name="run_full_evaluation")
    async def evaluate_response(
        self,
        user_message: str,
        bot_message: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on a bot response
        
        Args:
            user_message: User's input message
            bot_message: Bot's response message
            start_time: Optional start timestamp for latency evaluation
            end_time: Optional end timestamp for latency evaluation
            context: Additional context (user_id, session_id, etc.)
            
        Returns:
            Dict containing all evaluation results and aggregate scores
        """
        context = context or {}
        
        # Update LangFuse trace with context
        langfuse_context.update_current_trace(
            name="comprehensive_evaluation",
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            tags=["evaluation", "quality_check"],
            metadata={
                "user_message_length": len(user_message),
                "bot_message_length": len(bot_message),
                "has_latency_data": start_time is not None and end_time is not None
            }
        )
        
        try:
            # Create evaluation tasks for concurrent execution
            evaluation_tasks = []
            
            # LLM-based evaluations (run concurrently)
            evaluation_tasks.extend([
                self.empathy_evaluator.evaluate(user_message, bot_message, context),
                self.trigger_evaluator.evaluate(user_message, bot_message, context),
                self.crisis_evaluator.evaluate(user_message, bot_message, context),
                self.accessibility_evaluator.evaluate(user_message, bot_message, context)
            ])
            
            # Run LLM evaluations concurrently
            logger.info("Running concurrent evaluations...")
            empathy_result, trigger_result, crisis_result, accessibility_result = await asyncio.gather(
                *evaluation_tasks,
                return_exceptions=True
            )
            
            # Handle any exceptions from concurrent execution
            empathy_result = self._handle_evaluation_error(empathy_result, "empathy")
            trigger_result = self._handle_evaluation_error(trigger_result, "trigger")
            crisis_result = self._handle_evaluation_error(crisis_result, "crisis")
            accessibility_result = self._handle_evaluation_error(accessibility_result, "accessibility")
            
            # Run latency evaluation (if timestamps provided)
            latency_result = None
            if start_time is not None and end_time is not None:
                latency_result = await self.latency_evaluator.evaluate(start_time, end_time, context)
            
            # Compile results
            results = {
                "empathy": empathy_result,
                "trigger": trigger_result,
                "crisis": crisis_result,
                "accessibility": accessibility_result
            }
            
            if latency_result:
                results["latency"] = latency_result
            
            # Calculate aggregate scores
            aggregate_scores = self._calculate_aggregate_scores(results)
            results["aggregate_scores"] = aggregate_scores
            
            # Add metadata
            results["evaluation_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "evaluators_run": len([r for r in results.values() if not isinstance(r, dict) or "error" not in r]),
                "total_evaluators": 5 if latency_result else 4,
                "context": context
            }
            
            # Update LangFuse trace with results
            langfuse_context.update_current_observation(
                output=aggregate_scores,
                metadata={
                    "overall_quality_score": aggregate_scores.get("overall_quality_score", 0.0),
                    "success": True
                }
            )
            
            logger.info(f"Evaluation complete - Overall score: {aggregate_scores.get('overall_quality_score', 0.0):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            
            # Track error in LangFuse
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e),
                metadata={"success": False, "error": str(e)}
            )
            
            return {
                "error": str(e),
                "evaluation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                }
            }

    def _handle_evaluation_error(self, result: Any, evaluator_name: str) -> Dict[str, Any]:
        """Handle exceptions from concurrent evaluation execution"""
        if isinstance(result, Exception):
            logger.error(f"{evaluator_name} evaluation raised exception: {result}")
            return {
                "error": str(result),
                "evaluator": evaluator_name,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        return result

    def _calculate_aggregate_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate aggregate scores across all evaluations
        
        Returns:
            Dict with aggregate metrics
        """
        quality_scores = []
        score_breakdown = {}
        
        # Empathy score
        if "empathy" in results and isinstance(results["empathy"], dict):
            empathy_data = results["empathy"]
            if "average_empathy_score" in empathy_data:
                score = empathy_data["average_empathy_score"]
                quality_scores.append(score)
                score_breakdown["empathy_score"] = score
            elif "scores" in empathy_data and empathy_data["scores"]:
                score = statistics.mean(empathy_data["scores"].values())
                quality_scores.append(score)
                score_breakdown["empathy_score"] = score
        
        # Trigger detection (inverted - NO trigger is good)
        if "trigger" in results and isinstance(results["trigger"], dict):
            trigger_data = results["trigger"]
            if trigger_data.get("trigger_detected") == "NO":
                trigger_score = 1.0
            elif trigger_data.get("trigger_detected") == "YES":
                trigger_score = 0.0
            else:
                trigger_score = 0.5  # Unknown
            quality_scores.append(trigger_score)
            score_breakdown["trigger_safety_score"] = trigger_score
        
        # Crisis handling score
        if "crisis" in results and isinstance(results["crisis"], dict):
            crisis_data = results["crisis"]
            if "overall_score" in crisis_data and crisis_data["overall_score"] is not None:
                score = crisis_data["overall_score"]
                quality_scores.append(score)
                score_breakdown["crisis_safety_score"] = score
        
        # Accessibility score
        if "accessibility" in results and isinstance(results["accessibility"], dict):
            accessibility_data = results["accessibility"]
            if "accessibility_score" in accessibility_data:
                score = accessibility_data["accessibility_score"]
                quality_scores.append(score)
                score_breakdown["accessibility_score"] = score
        
        # Latency performance score
        if "latency" in results and isinstance(results["latency"], dict):
            latency_data = results["latency"]
            if "performance_score" in latency_data:
                score = latency_data["performance_score"]
                quality_scores.append(score)
                score_breakdown["performance_score"] = score
        
        # Calculate overall quality score
        overall_score = statistics.mean(quality_scores) if quality_scores else 0.0
        
        return {
            "overall_quality_score": round(overall_score, 3),
            "score_breakdown": score_breakdown,
            "metrics_evaluated": len(quality_scores),
            "quality_level": self._get_quality_level(overall_score)
        }

    def _get_quality_level(self, score: float) -> str:
        """Categorize quality score into levels"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        else:
            return "needs_improvement"

    def get_latency_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for latency evaluator"""
        return self.latency_evaluator.get_summary_statistics()

    def reset_latency_statistics(self):
        """Reset latency statistics"""
        self.latency_evaluator.reset_statistics()

