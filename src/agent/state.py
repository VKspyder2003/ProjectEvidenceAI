from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add
from pydantic import BaseModel, Field
from enum import Enum

class FailureType(str, Enum):
    NONE = "none"
    RECOVERABLE = "recoverable"
    TRANSIENT = "transient"
    FATAL = "fatal"

class SessionContext(BaseModel):
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None
    default_branch: Optional[str] = None

class PlanStep(BaseModel):
    id: int = Field(..., description="Unique step identifier")
    tool_name: str = Field(..., description="The name of the tool to call")
    arguments: Dict[str, Any] = Field(
        ..., description="Arguments for the tool"
    )
    reason: str = Field(
        ..., description="Reason for executing this step"
    )

class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(
        ...,
        description="Ordered list of tool execution steps"
    )

class ToolCallRecord(BaseModel):
    step_id: int = Field(
        ..., description="The ID of the related plan step"
    )
    tool_name: str = Field(
        ..., description="Name of the tool executed"
    )
    arguments: Dict[str, Any] = Field(
        ..., description="Arguments used for execution"
    )
    result: Any = Field(
        ..., description="Structured result returned by the tool"
    )
    failure_type: FailureType = Field(
        default=FailureType.NONE, description="Classification of failure if the step failed"
    )

class Evidence(BaseModel):
    source_type: str = Field(
        ..., description="Evidence type: pull_request, issue, or file"
    )
    source_id: str = Field(
        ..., description="Unique GitHub source identifier"
    )
    url: Optional[str] = Field(
        default=None,
        description="GitHub URL if available",
    )
    content: Dict[str, Any] = Field(
        ..., description="Relevant content extracted from the source"
    )
    token_estimate: int = Field(
        default=0, description="Approximate token count of the content"
    )

class AgentState(TypedDict):
    # User intent — never mutate
    query: str

    # Planning & Scoping
    plan: List[PlanStep]
    current_step: int
    plan_version: int

    # Execution trace
    tool_calls_history: Annotated[List[ToolCallRecord], add]
    last_failure: Optional[Dict[str, Any]]

    # Evidence
    retrieved_evidence: Annotated[List[Evidence], add]
    budgeted_evidence: Optional[List[Evidence]]
    budget_consumed: int

    # Recovery
    retry_count: int
    correction_hints: Annotated[List[str], add]
    failed_step_id: Optional[int]

    # Session context
    session_context: SessionContext

    # Output & Final State
    draft_response: Optional[str]
    output_validation_result: Optional[Dict[str, Any]]
    fatal_error: bool
    error: Optional[str]
