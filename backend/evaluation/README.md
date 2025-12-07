# ThinkLife Evaluation System

Comprehensive evaluation framework for AI responses with LangFuse observability, tracing, and monitoring.

## Overview

The evaluation system provides trauma-informed assessment of AI responses across multiple dimensions:

- **Empathy Assessment**: Evaluates emotional understanding and validation
- **Trigger Detection**: Identifies potentially triggering language
- **Crisis Handling**: Assesses safety and trauma-informed care
- **Latency Monitoring**: Tracks response time performance
- **Accessibility**: Evaluates language clarity and accessibility

## Architecture

```
evaluation/
├── evaluation.py              # Main entry point with LangFuse integration
├── evaluators/
│   ├── __init__.py
│   ├── empathy_evaluator.py
│   ├── trigger_evaluator.py
│   ├── crisis_evaluator.py
│   ├── latency_evaluator.py
│   ├── accessibility_evaluator.py
│   └── evaluation_manager.py  # Coordinates all evaluators
└── README.md
```

## Features

### LangFuse Observability

Every evaluation is automatically traced in LangFuse with:
- **Tracing**: Track individual evaluator calls and LLM judgments
- **Monitoring**: Monitor evaluation performance over time
- **Metrics**: Aggregate quality scores and trends
- **Session Tracking**: Link evaluations to user sessions
- **Error Tracking**: Capture and log evaluation failures

### Evaluators

#### 1. Empathy Evaluator
Assesses emotional understanding across 5 dimensions:
- Validation of feelings
- Perspective-taking
- Tone sensitivity
- Politeness and soft language
- Encouragement and hopefulness

#### 2. Trigger Evaluator
Detects triggering language using:
- Few-shot learning examples
- Pattern recognition
- Confidence scoring
- Specific trigger phrase identification

#### 3. Crisis Evaluator
Evaluates 20 safety aspects including:
- Professional help encouragement
- Safety planning
- Non-judgmental phrasing
- Appropriate scope
- Emergency resource provision

#### 4. Latency Evaluator
Monitors performance with:
- Response time tracking
- Statistical analysis (mean, median, p95, p99)
- Performance categorization
- Trend analysis

#### 5. Accessibility Evaluator
Assesses language clarity:
- Readability
- Complexity level
- Tone appropriateness
- Cultural sensitivity

## Setup

### 1. Environment Variables

Add to your `.env` file:

```bash
# LangFuse Configuration (Required for observability)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, defaults to cloud

# LLM Provider (Required for LLM-as-judge evaluations)
OPENAI_API_KEY=your_openai_key
# OR
GOOGLE_API_KEY=your_gemini_key
```

### 2. Get LangFuse Keys

1. Sign up at https://cloud.langfuse.com
2. Create a new project
3. Copy your public and secret keys
4. Add them to your `.env` file

## Usage

### Basic Usage

```python
from evaluation import evaluate_response
import time

# Time the response
start_time = time.time()
bot_response = await generate_bot_response(user_message)
end_time = time.time()

# Run comprehensive evaluation
results = await evaluate_response(
    user_message="I'm feeling really anxious today",
    bot_message=bot_response,
    start_time=start_time,
    end_time=end_time,
    context={
        "user_id": "user_123",
        "session_id": "session_456",
        "ace_score": 3  # Optional context
    }
)

# Access results
print(f"Overall Quality: {results['aggregate_scores']['overall_quality_score']}")
print(f"Empathy Score: {results['empathy']['average_empathy_score']}")
print(f"Trigger Detected: {results['trigger']['trigger_detected']}")
print(f"Crisis Safety: {results['crisis']['overall_score']}")
```

### Using Individual Evaluators

```python
from evaluation import get_evaluation_manager

# Get the manager
manager = get_evaluation_manager()

# Run specific evaluations
empathy_result = await manager.empathy_evaluator.evaluate(
    user_message="I'm struggling",
    bot_message="I hear you...",
    context={"user_id": "user_123"}
)

trigger_result = await manager.trigger_evaluator.evaluate(
    user_message="I'm worried",
    bot_message="Don't worry, be happy!",
    context={"user_id": "user_123"}
)
```

### Initialize with Custom LLM Client

```python
from evaluation import initialize_evaluation_manager
from brain import CortexFlow

# Get your LLM client
brain = CortexFlow()
llm_client = brain.providers.get("gemini")

# Initialize with LLM client
manager = initialize_evaluation_manager(llm_client=llm_client)

# Now run evaluations
results = await manager.evaluate_response(
    user_message="...",
    bot_message="..."
)
```

### Get Latency Statistics

```python
from evaluation import get_latency_statistics, reset_latency_statistics

# Get comprehensive latency stats
stats = get_latency_statistics()
print(f"Average Latency: {stats['average_latency_recent']}s")
print(f"P95 Latency: {stats['p95_latency']}s")
print(f"Performance Distribution: {stats['performance_distribution']}")

# Reset statistics
reset_latency_statistics()
```

## Response Format

### Full Evaluation Results

```python
{
    "empathy": {
        "scores": {
            "validation_of_feelings": 0.9,
            "perspective_taking": 0.85,
            "tone_sensitivity": 0.95,
            "politeness_soft_language": 0.9,
            "encouragement_hopefulness": 0.8
        },
        "reasoning": {...},
        "average_empathy_score": 0.88,
        "evaluation_timestamp": "2024-01-15T10:30:00Z"
    },
    "trigger": {
        "trigger_detected": "NO",
        "confidence": 0.95,
        "reasoning": "Response uses validating, supportive language",
        "trigger_phrases": [],
        "evaluation_timestamp": "2024-01-15T10:30:00Z"
    },
    "crisis": {
        "aspect_scores": {
            "encourages_professional_help": 1.0,
            "avoids_unsafe_illegal_advice": 1.0,
            ...
        },
        "overall_score": 0.92,
        "safety_concerns": [],
        "strengths": ["Validates feelings", "Suggests resources"],
        "evaluation_timestamp": "2024-01-15T10:30:00Z"
    },
    "accessibility": {
        "accessibility_score": 0.9,
        "clarity_score": 0.95,
        "complexity_score": 0.85,
        "tone_appropriateness_score": 0.9,
        "suggestions": [],
        "evaluation_timestamp": "2024-01-15T10:30:00Z"
    },
    "latency": {
        "response_latency_seconds": 2.3,
        "performance_score": 0.8,
        "performance_category": "good",
        "statistics": {...},
        "evaluation_timestamp": "2024-01-15T10:30:00Z"
    },
    "aggregate_scores": {
        "overall_quality_score": 0.89,
        "score_breakdown": {
            "empathy_score": 0.88,
            "trigger_safety_score": 1.0,
            "crisis_safety_score": 0.92,
            "accessibility_score": 0.9,
            "performance_score": 0.8
        },
        "metrics_evaluated": 5,
        "quality_level": "good"
    },
    "evaluation_metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "evaluators_run": 5,
        "total_evaluators": 5,
        "context": {...}
    }
}
```

## LangFuse Dashboard

View your evaluations in LangFuse:

1. **Traces**: See each evaluation call with timing
2. **Sessions**: Group evaluations by user session
3. **Scores**: Track quality metrics over time
4. **Analytics**: Monitor trends and patterns
5. **Errors**: Debug failed evaluations

## Best Practices

### 1. Always Provide Context

```python
results = await evaluate_response(
    user_message=msg,
    bot_message=response,
    context={
        "user_id": user_id,
        "session_id": session_id,
        "ace_score": ace_score,  # Important for trauma-informed evaluation
        "agent_type": "zoe"
    }
)
```

### 2. Track Latency

```python
import time

start = time.time()
response = await agent.process(message)
end = time.time()

results = await evaluate_response(
    user_message=message,
    bot_message=response,
    start_time=start,
    end_time=end
)
```

### 3. Monitor Quality Trends

Check LangFuse dashboard regularly for:
- Quality score trends
- Common trigger patterns
- Performance degradation
- User session patterns

### 4. Act on Results

```python
results = await evaluate_response(...)

# Check for safety concerns
if results["trigger"]["trigger_detected"] == "YES":
    logger.warning("Trigger detected in response")
    # Take action: log, alert, or regenerate

# Check quality threshold
if results["aggregate_scores"]["overall_quality_score"] < 0.7:
    logger.warning("Low quality response detected")
    # Consider regenerating response
```

## Extending

### Add New Evaluator

1. Create `evaluators/my_evaluator.py`:

```python
from langfuse.decorators import observe, langfuse_context

class MyEvaluator:
    @observe(name="my_evaluation")
    async def evaluate(self, user_message, bot_message, context=None):
        # Your evaluation logic
        result = {...}
        
        langfuse_context.update_current_observation(
            output=result,
            metadata={"success": True}
        )
        
        return result
```

2. Add to `evaluators/__init__.py`
3. Add to `EvaluationManager` in `evaluation_manager.py`

## Troubleshooting

### LangFuse Not Working

- Check environment variables are set
- Verify API keys are correct
- Check network connectivity to LangFuse host
- Review LangFuse logs for errors

### Evaluation Timeouts

- Increase LLM timeout settings
- Check LLM API status
- Monitor network latency

### Low Quality Scores

- Review bot responses for common issues
- Check trigger detection results
- Analyze empathy scores by dimension
- Review LangFuse traces for patterns

## License

Part of ThinkLife AI System - © 2024 Think Round, Inc.

