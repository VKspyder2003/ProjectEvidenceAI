from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from src.agent.state import AgentState
from src.guardrails.output_validator import validate_output

async def output_validator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Validates the drafted response to ensure all citations map strictly
    to the retrieved evidence. Provides a safe fallback if violations occur.
    """
    draft_response = state.get("draft_response")
    retrieved_evidence = state.get("retrieved_evidence", [])
    
    if not draft_response:
        return {}

    result = validate_output(draft_response, retrieved_evidence)
    
    # We must store the result struct. It's a Pydantic model so we dump it to dict.
    updates: Dict[str, Any] = {
        "output_validation_result": result.model_dump()
    }
    
    if not result.valid:
        # Replace the unsafe/unverified draft with a safe fallback
        updates["draft_response"] = "The response could not be fully verified against retrieved evidence."
        
    return updates
