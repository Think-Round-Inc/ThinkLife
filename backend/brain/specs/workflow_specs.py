"""
Workflow specifications and types
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class WorkflowStatus(str, Enum):
    """Status of workflow execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    ESCALATED = "escalated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowType(str, Enum):
    """Type of workflow"""
    CONVERSATIONAL = "conversational"
    TASK_BASED = "task_based"
    ORCHESTRATED = "orchestrated"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """A single step in the workflow"""
    step_id: str
    name: str
    action: str  # "reason", "query_data", "use_tool", "call_provider"
    params: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    timeout: float = 30.0
    retry_count: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# NOTE: a first `WorkflowExecution` dataclass used to live here and was silently
# shadowed by the one at the bottom of this file. It was removed (US 1.1); the
# surviving definition below is the one the rest of the code always used.


@dataclass
class ProcessingSpec:
    """Specification for how agents want the request to be processed"""
    max_iterations: int = 3
    timeout_seconds: float = 30.0
    enable_context_enhancement: bool = True
    enable_safety_checks: bool = True
    enable_conversation_memory: bool = True
    stream_response: bool = False
    
    # Two-Phase Execution Settings
    execution_strategy: str = "direct"  # direct | reasoned | adaptive
    reasoning_threshold: float = 0.7  # Minimum confidence to use reasoning suggestions
    
    # Evaluation and Observability
    eval: bool = False  # Enable evaluation when True
    
    custom_processing: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "enable_context_enhancement": self.enable_context_enhancement,
            "enable_safety_checks": self.enable_safety_checks,
            "enable_conversation_memory": self.enable_conversation_memory,
            "stream_response": self.stream_response,
            "execution_strategy": self.execution_strategy,
            "reasoning_threshold": self.reasoning_threshold,
            "eval": self.eval,
            **self.custom_processing
        }


@dataclass
class ExecutionPlan:
    """
    Output of planning phase - contains the final execution plan
    
    This is created by CortexFlow during the planning phase and passed to
    the workflow engine for execution.
    """
    # The original specs from the agent
    original_specs: Any  # AgentExecutionSpec
    
    # The specs to actually execute (may be optimized by reasoning)
    optimized_specs: Any  # AgentExecutionSpec
    
    # Whether reasoning was applied
    reasoning_applied: bool
    
    # Reasoning confidence
    confidence: float
    
    # Reasoning notes and suggestions
    reasoning_notes: Dict[str, Any] = field(default_factory=dict)
    
    # Estimated execution cost
    estimated_cost: float = 0.0
    
    # Estimated latency in seconds
    estimated_latency: float = 0.0
    
    # Planning timestamp
    planned_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "reasoning_applied": self.reasoning_applied,
            "confidence": self.confidence,
            "reasoning_notes": self.reasoning_notes,
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency,
            "planned_at": self.planned_at.isoformat()
        }


@dataclass
class WorkflowExecution:
    """Information about workflow execution"""
    execution_id: str
    workflow_type: WorkflowType
    agent_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    steps_executed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None



@dataclass
class WorkflowProfile:
    """
    Per-agent settings for the standard LangGraph workflow (US 1.1).

    Every agent (Zoe, FAQ, Bard, ...) runs the same graph; the profile decides
    the limits and the failure behaviour. Built-in profiles live in
    `brain/cortex/workflow_profiles.py`; an agent's own `ProcessingSpec`
    (max_iterations, timeout_seconds, enable_safety_checks) overrides them.
    """
    agent_type: str = "default"

    # Quality loop: total answer attempts (first try included) before escalating
    max_iterations: int = 3
    confidence_threshold: float = 0.75
    llm_validation: bool = True  # second LLM call that scores each answer

    # Time budget. No new work (retry, fallback) starts after run_timeout_seconds;
    # an in-flight provider call may finish, bounded by provider_timeout_seconds.
    run_timeout_seconds: float = 60.0
    provider_timeout_seconds: float = 30.0

    # Transient-error retries (timeouts, 429, 5xx, dropped connections)
    provider_retry_attempts: int = 3
    data_source_retry_attempts: int = 2
    retry_initial_interval: float = 1.0
    retry_max_interval: float = 10.0
    retry_jitter: bool = True  # adds up to 25% random delay per retry

    # Output policy checks (brain/guardrails/policy_validator.py)
    safety_checks: bool = True

    # Tried in order, once each, when the primary provider is still failing
    # after its retries. Items are ProviderSpec objects.
    fallback_providers: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "max_iterations": self.max_iterations,
            "confidence_threshold": self.confidence_threshold,
            "llm_validation": self.llm_validation,
            "run_timeout_seconds": self.run_timeout_seconds,
            "provider_timeout_seconds": self.provider_timeout_seconds,
            "provider_retry_attempts": self.provider_retry_attempts,
            "data_source_retry_attempts": self.data_source_retry_attempts,
            "safety_checks": self.safety_checks,
            "fallback_providers": [
                f"{p.provider_type}:{p.model}" for p in self.fallback_providers
            ],
        }
