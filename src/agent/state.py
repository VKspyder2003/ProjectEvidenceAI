from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add
from pydantic import BaseModel, Field

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

class AgentState(TypedDict):
    query: str
    plan: List[PlanStep]
    current_step: int

    tool_calls_history: Annotated[
        List[ToolCallRecord],
        add,
    ]

    retrieved_evidence: Annotated[
        List[Evidence],
        add,
    ]

    retry_count: int
    draft_response: Optional[str]
    error: Optional[str]
