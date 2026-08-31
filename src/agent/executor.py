from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from fastmcp import Client

from .state import AgentState, ToolCallRecord, Evidence, FailureType
from .dependencies import AgentDependencies

def classify_failure(tool_name: str, status_code: int, message: str) -> FailureType:
    """Classify the failure based on status code, message, and tool."""
    if status_code == 401:
        return FailureType.FATAL
    elif status_code == 403:
        if "rate limit" in message.lower():
            return FailureType.TRANSIENT
        return FailureType.RECOVERABLE
    elif status_code == 422:
        if "validation failed" in message.lower() and tool_name == "search_issues":
            return FailureType.FATAL
        return FailureType.RECOVERABLE
    elif status_code == 429:
        return FailureType.TRANSIENT
    elif status_code == 404:
        if "not found" in message.lower() and tool_name != "read_repository_file":
            # Very likely missing repository if a search or PR fetch fails with 404
            return FailureType.FATAL
        return FailureType.RECOVERABLE
    return FailureType.RECOVERABLE


async def executor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executor node that executes the next tool call in the plan.
    It executes exactly one step from the plan based on current_step.
    """
    configurable = config.get("configurable", {})
    dependencies: AgentDependencies = configurable.get("dependencies")
    
    if not dependencies:
        raise ValueError("AgentDependencies not found in RunnableConfig['configurable']")

    mcp_server = dependencies.mcp_server
    
    plan = state.get("plan", [])
    current_step_index = state.get("current_step", 0)
    
    if current_step_index >= len(plan):
        # Nothing left to execute
        return {}

    step = plan[current_step_index]
    
    new_evidence: List[Evidence] = []
    tool_history: List[ToolCallRecord] = []
    
    try:
        async with Client(mcp_server) as client:
            result = await client.call_tool(step.tool_name, step.arguments)
            content = result.structured_content
            
            # Normalize results into evidence
            if isinstance(content, dict) and content.get("success"):
                data = content.get("data")
                
                def estimate_tokens(obj):
                    return len(str(obj)) // 4
                
                if step.tool_name == "get_recent_pull_requests":
                    for pr in data:
                        new_evidence.append(
                            Evidence(
                                source_type="pull_request",
                                source_id=f"PR-{pr.get('number', 'unknown')}",
                                url=pr.get("html_url"),
                                content=pr,
                                token_estimate=estimate_tokens(pr)
                            )
                        )
                elif step.tool_name == "get_pr_diff":
                    pr_num = data.get("pr_number", "unknown")
                    new_evidence.append(
                        Evidence(
                            source_type="pull_request_diff",
                            source_id=f"PR-{pr_num}-diff",
                            content=data,
                            token_estimate=estimate_tokens(data)
                        )
                    )
                elif step.tool_name == "search_issues":
                    for issue in data:
                        new_evidence.append(
                            Evidence(
                                source_type="issue",
                                source_id=f"Issue-{issue.get('number', 'unknown')}",
                                url=issue.get("html_url"),
                                content=issue,
                                token_estimate=estimate_tokens(issue)
                            )
                        )
                elif step.tool_name == "read_repository_file":
                    path = data.get("file_path", "unknown")
                    new_evidence.append(
                        Evidence(
                            source_type="file",
                            source_id=path,
                            url=data.get("html_url"),
                            content=data,
                            token_estimate=estimate_tokens(data)
                        )
                    )
                else:
                    new_evidence.append(
                        Evidence(
                            source_type="unknown",
                            source_id=f"{step.tool_name}-{step.id}",
                            content={"data": data},
                            token_estimate=estimate_tokens(data)
                        )
                    )

            
            if isinstance(content, dict) and not content.get("success"):
                err = content.get("error", {})
                status_code = err.get("status_code", 500)
                msg = err.get("message", "Unknown error")
                failure_type = classify_failure(step.tool_name, status_code, msg)
                
                record = ToolCallRecord(
                    step_id=step.id,
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    result=content,
                    failure_type=failure_type
                )
                tool_history.append(record)
                
                # Update last_failure
                state["last_failure"] = {
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "error": err,
                    "failure_type": failure_type
                }
            else:
                tool_history.append(
                    ToolCallRecord(
                        step_id=step.id,
                        tool_name=step.tool_name,
                        arguments=step.arguments,
                        result=content,
                        failure_type=FailureType.NONE
                    )
                )

    except Exception as e:
        # Record failure cleanly for an unhandled exception
        failure_type = FailureType.RECOVERABLE
        record = ToolCallRecord(
            step_id=step.id,
            tool_name=step.tool_name,
            arguments=step.arguments,
            result={"success": False, "error": str(e)},
            failure_type=failure_type
        )
        tool_history.append(record)
        state["last_failure"] = {
            "tool_name": step.tool_name,
            "arguments": step.arguments,
            "error": {"message": str(e), "status_code": 500},
            "failure_type": failure_type
        }
        
    budget_consumed = state.get("budget_consumed", 0)
    for ev in new_evidence:
        budget_consumed += ev.token_estimate
        
    updates = {
        "tool_calls_history": tool_history,
        "retrieved_evidence": new_evidence,
        "current_step": current_step_index + 1,
        "budget_consumed": budget_consumed
    }
    
    if "last_failure" in state:
        updates["last_failure"] = state["last_failure"]
        
    return updates
