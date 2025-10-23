from evaluation.llm_evaluators import TraumaAwareEvaluator, LanguageAccessibilityEvaluator
from evaluation.objective_metrics import ObjectiveMetricsEvaluator
from brain.brain_core import ThinkxLifeBrain
import asyncio
import json
import statistics
from datetime import datetime

# Global evaluator instances
trauma_evaluator = TraumaAwareEvaluator()
accessibility_evaluator = LanguageAccessibilityEvaluator()
objective_evaluator = ObjectiveMetricsEvaluator()

# Helper function for subjective metrics (LLM-judged)
async def run_subjective_evaluation(user_message: str, bot_message: str):
    """
    Run subjective evaluations (empathy, trigger, crisis, accessibility) on a bot response.
    """
    evaluation_tasks = [
        asyncio.create_task(trauma_evaluator.evaluate_empathy(user_message, bot_message)),
        asyncio.create_task(trauma_evaluator.evaluate_trigger(user_message, bot_message)),
        asyncio.create_task(trauma_evaluator.evaluate_crisis(user_message, bot_message)),
        asyncio.create_task(accessibility_evaluator.evaluate_accessibility(user_message, bot_message))
    ]

    empathy_result, trigger_result, crisis_result, accessibility_result = await asyncio.gather(*evaluation_tasks)

    results = {
        "empathy": empathy_result,
        "trigger": trigger_result,
        "crisis": crisis_result,
        "accessibility": accessibility_result
    }

    # Calculate overall subjective score
    quality_scores = []

    if isinstance(empathy_result, dict) and "scores" in empathy_result:
        empathy_scores = empathy_result["scores"]
        if empathy_scores:
            quality_scores.append(statistics.mean(empathy_scores.values()))

    if isinstance(crisis_result, dict) and "overall_score" in crisis_result and crisis_result["overall_score"] is not None:
        quality_scores.append(crisis_result["overall_score"])

    if isinstance(accessibility_result, dict) and "accessibility_score" in accessibility_result:
        quality_scores.append(accessibility_result["accessibility_score"])

    overall_subjective_score = statistics.mean(quality_scores) if quality_scores else 0.0
    results["overall_subjective_score"] = overall_subjective_score
    results["subjective_metrics_evaluated"] = len(quality_scores)
    results["evaluation_timestamp"] = datetime.now().isoformat()

    return results

# Helper function for objective metrics (computable: relevance + latency)
async def run_objective_evaluation(user_message: str, bot_message: str, start_time: float = None, end_time: float = None):
    """
    Run objective evaluations (relevance, latency) on a bot response.
    """
    # Relevance evaluation
    relevance_result = await objective_evaluator.evaluate_relevance(user_message, bot_message)

    # Latency evaluation (only if timestamps provided)
    latency_result = None
    if start_time is not None and end_time is not None:
        latency_result = await objective_evaluator.evaluate_latency(start_time, end_time)

    results = {
        "relevance": relevance_result
    }

    if latency_result:
        results["latency"] = latency_result

    # Calculate overall objective score
    quality_scores = []

    if isinstance(relevance_result, dict) and "relevance_score" in relevance_result:
        quality_scores.append(relevance_result["relevance_score"])

    if latency_result and "performance_score" in latency_result:
        quality_scores.append(latency_result["performance_score"])

    overall_objective_score = statistics.mean(quality_scores) if quality_scores else 0.0
    results["overall_objective_score"] = overall_objective_score
    results["objective_metrics_evaluated"] = len(quality_scores)
    results["evaluation_timestamp"] = datetime.now().isoformat()

    return results

# Combined helper function (original run_evaluation)
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

    # Calculate combined overall score
    quality_scores = []

    if "overall_subjective_score" in subjective_results:
        quality_scores.append(subjective_results["overall_subjective_score"])

    if "overall_objective_score" in objective_results:
        quality_scores.append(objective_results["overall_objective_score"])

    overall_score = statistics.mean(quality_scores) if quality_scores else 0.0
    results["overall_quality_score"] = overall_score
    results["metrics_evaluated"] = subjective_results.get("subjective_metrics_evaluated", 0) + objective_results.get("objective_metrics_evaluated", 0)
    results["evaluation_timestamp"] = datetime.now().isoformat()

    return results