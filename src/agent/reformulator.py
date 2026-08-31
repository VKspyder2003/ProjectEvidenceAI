import json
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState
from .prompts import REFORMULATOR_SYSTEM_PROMPT
from .dependencies import AgentDependencies

async def reformulator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Reformulator node that analyzes a failure and generates a correction hint.
    """
    configurable = config.get("configurable", {})
    dependencies: AgentDependencies = configurable.get("dependencies")
    
    if not dependencies:
        raise ValueError("AgentDependencies not found in config")

    llm = dependencies.llm
    
    last_failure = state.get("last_failure")
    if not last_failure:
        return {"correction_hints": ["Unknown failure occurred. Try a different approach."]}

    tool_history = state.get("tool_calls_history", [])
    if tool_history:
        history_str = "\n".join([
            f"Step {s.step_id}: {s.tool_name}({json.dumps(s.arguments)}) -> Success: {s.failure_type.value == 'none'}" 
            for s in tool_history
        ])
    else:
        history_str = "No previous executions."

    system_prompt = REFORMULATOR_SYSTEM_PROMPT.format(
        history=history_str,
        tool_name=last_failure.get("tool_name", "Unknown"),
        arguments=json.dumps(last_failure.get("arguments", {})),
        error=json.dumps(last_failure.get("error", {}))
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Provide a concise correction hint.")
    ]

    try:
        response = await llm.ainvoke(messages)
        hint = str(response.content).strip()
    except Exception as e:
        hint = f"Error generating hint: {e}"

    # Increment retry count, add hint, mark the step that failed
    retry_count = state.get("retry_count", 0) + 1
    
    # We get the failed step ID from the tool call history
    failed_step_id = None
    if tool_history:
        failed_step_id = tool_history[-1].step_id
        
    hints = state.get("correction_hints", []) + [hint]

    return {
        "correction_hints": hints,
        "retry_count": retry_count,
        "failed_step_id": failed_step_id
    }
