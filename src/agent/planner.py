import json
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from fastmcp import Client

from .state import AgentState, ExecutionPlan
from .prompts import PLANNER_SYSTEM_PROMPT
from .dependencies import AgentDependencies

async def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Planner node that selects MCP tools to execute based on the user's query.
    It does not execute tools, it only outputs the plan.
    """
    # 1 & 2. Receive query and dependencies
    query = state.get("query", "")
    session_context = state.get("session_context")
    last_failure = state.get("last_failure")
    correction_hints = state.get("correction_hints", [])
    
    old_plan = state.get("plan", [])
    failed_step_id = state.get("failed_step_id")
    
    successful_steps_list = []
    if failed_step_id is not None:
        successful_steps_list = [s for s in old_plan if s.id < failed_step_id]
    else:
        # If there's no failure but an old plan exists, use completed steps
        successful_steps_list = [s for s in old_plan if s.id < state.get("current_step", 0)]
        
    session_context_str = json.dumps(session_context.model_dump()) if session_context else "None"
    last_failure_str = json.dumps(last_failure) if last_failure else "None"
    correction_hints_str = "\n".join(f"- {h}" for h in correction_hints) if correction_hints else "None"
    successful_steps_str = "\n".join(f"- Step {s.id}: {s.tool_name}({s.arguments})" for s in successful_steps_list) if successful_steps_list else "None"

    
    configurable = config.get("configurable", {})
    dependencies: AgentDependencies = configurable.get("dependencies")
    
    if not dependencies:
        raise ValueError("AgentDependencies not found in RunnableConfig['configurable']")

    llm = dependencies.llm
    mcp_server = dependencies.mcp_server

    # 3. Load available MCP tool schemas/descriptions
    # We open a brief client session strictly to list tools, not to execute them.
    async with Client(mcp_server) as client:
        tools = await client.list_tools()

    # 4. Format tool descriptions
    tool_descriptions = []
    allowed_tools = set()
    for t in tools:
        allowed_tools.add(t.name)
        desc = f"Tool: {t.name}\nDescription: {t.description}\nSchema: {json.dumps(t.inputSchema)}\n"
        tool_descriptions.append(desc)
    
    formatted_tool_descriptions = "\n".join(tool_descriptions)

    session_str = f"Repo: {session_context.repo_owner}/{session_context.repo_name}" if session_context else "None"
    hints_str = "\n".join(f"- {h}" for h in correction_hints) if correction_hints else "None"
    
    # Format tool history for context
    tool_history = state.get("tool_calls_history", [])
    history_str = "None"
    if tool_history:
        history_lines = []
        for r in tool_history:
            status = "Success" if (r.result and isinstance(r.result, dict) and r.result.get("success")) else "Failed"
            history_lines.append(f"- Step {r.step_id}: {r.tool_name}({r.arguments}) -> {status}")
            if status == "Failed":
                err = r.result.get("error") if isinstance(r.result, dict) else "Unknown"
                history_lines.append(f"  Error: {err}")
        history_str = "\n".join(history_lines)

    # Inject into prompt
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        tool_descriptions=formatted_tool_descriptions,
        successful_steps=successful_steps_str,
        history=history_str,
        last_failure=json.dumps(last_failure) if last_failure else "None",
        correction_hints=hints_str,
        session_context=session_str
    )

    # 5. Force structured LLM output into ExecutionPlan
    structured_llm = llm.with_structured_output(ExecutionPlan)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    try:
        plan: ExecutionPlan = await structured_llm.ainvoke(messages)
    except Exception as e:
        return {"error": f"Planner failed to generate a valid plan: {str(e)}"}

    # 6. Validate output against allowlist and merge
    valid_steps = []
    
    # Keep successful steps
    for s in successful_steps_list:
        valid_steps.append(s)
        
    next_id = len(valid_steps)
    
    # In recovery, the first generated step REPLACES the failed step
    # We defensively filter exact duplicates of last_failure
    is_recovery = failed_step_id is not None
    
    generated_steps = []
    
    for step in plan.steps:
        if step.tool_name not in allowed_tools:
            return {"error": f"Planner generated invalid tool name: {step.tool_name}"}
        
        # Defensive backstop: filter exact duplicate of any previous failure
        if is_recovery:
            is_duplicate = False
            for past_call in state.get("tool_calls_history", []):
                if past_call.result and isinstance(past_call.result, dict) and not past_call.result.get("success"):
                    if step.tool_name == past_call.tool_name and step.arguments == past_call.arguments:
                        is_duplicate = True
                        break
            if is_duplicate:
                continue
                
        step.id = next_id
        valid_steps.append(step)
        generated_steps.append(step)
        next_id += 1
        
    # If this is a recovery, and the planner only provided the replacement for the failed step,
    # we need to append any remaining unexecuted steps from the original plan.
    if is_recovery and failed_step_id is not None:
        # Check if the planner already included the remaining steps.
        # We assume if it only generated 1 step, it only generated the replacement.
        # Or more robustly, just use whatever it gave as the new tail.
        # The prompt tells it to "Output ONLY the replacement step(s) and any remaining unexecuted steps."
        # So we trust `valid_steps` as the complete plan.
        pass

        
    # 7. Return partial state updates
    return {
        "plan": valid_steps,
        "current_step": len(successful_steps_list),
        "plan_version": state.get("plan_version", 0) + 1,
        "error": None
    }
