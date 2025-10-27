from evaluation.metrics.empathy_evaluator import EmpathyEvaluator
from evaluation.metrics.trigger_evaluator import TriggerEvaluator
from evaluation.metrics.crisis_evaluator import CrisisEvaluator
from evaluation.metrics.accessibility_evaluator import AccessibilityEvaluator
from evaluation.metrics.relevance_evaluator import RelevanceEvaluator
from evaluation.metrics.latency_evaluator import LatencyEvaluator
import asyncio
import json
import statistics
from datetime import datetime

# Global evaluator instances
empathy_evaluator = EmpathyEvaluator()
trigger_evaluator = TriggerEvaluator()
crisis_evaluator = CrisisEvaluator()
accessibility_evaluator = AccessibilityEvaluator()
relevance_evaluator = RelevanceEvaluator()
latency_evaluator = LatencyEvaluator()

# Helper function for subjective metrics 
async def run_subjective_evaluation(user_message: str, bot_message: str):
    """
    Run subjective evaluations (empathy, trigger, crisis, accessibility) on a bot response.
    """
    evaluation_tasks = [
        asyncio.create_task(empathy_evaluator.evaluate(user_message, bot_message)),
        asyncio.create_task(trigger_evaluator.evaluate(user_message, bot_message)),
        asyncio.create_task(crisis_evaluator.evaluate(user_message, bot_message)),
        asyncio.create_task(accessibility_evaluator.evaluate(user_message, bot_message))
    ]
    empathy_result, trigger_result, crisis_result, accessibility_result = await asyncio.gather(*evaluation_tasks)
    results = {
        "empathy": empathy_result,
        "trigger": trigger_result,
        "crisis": crisis_result,
        "accessibility": accessibility_result
    }
    
    results["evaluation_timestamp"] = datetime.now().isoformat()
    return results

# Helper function for objective metrics 
async def run_objective_evaluation(user_message: str, bot_message: str, start_time: float = None, end_time: float = None):
    """
    Run objective evaluations (relevance, latency) on a bot response.
    """
    
    relevance_result = await relevance_evaluator.evaluate(user_message, bot_message)
    
    latency_result = None
    if start_time is not None and end_time is not None:
        latency_result = await latency_evaluator.evaluate(start_time, end_time)
    results = {
        "relevance": relevance_result
    }
    if latency_result:
        results["latency"] = latency_result
    
    results["evaluation_timestamp"] = datetime.now().isoformat()
    return results

# Combined helper function 
async def run_evaluation(user_message: str, bot_message: str, start_time: float = None, end_time: float = None):
    """
    Run all evaluations (both subjective and objective) on a bot response.
    """
    subjective_results = await run_subjective_evaluation(user_message, bot_message)
    objective_results = await run_objective_evaluation(user_message, bot_message, start_time, end_time)
    results = {
        "subjective": subjective_results,
        "objective": objective_results
    }
    results["evaluation_timestamp"] = datetime.now().isoformat()
    return results