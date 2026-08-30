from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from fastmcp import Client

from .state import AgentState, ToolCallRecord, Evidence
from .dependencies import AgentDependencies

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
                
                if step.tool_name == "get_recent_pull_requests":
                    for pr in data:
                        new_evidence.append(
                            Evidence(
                                source_type="pull_request",
                                source_id=f"PR-{pr.get('number', 'unknown')}",
                                url=pr.get("html_url"),
                                content=pr
                            )
                        )
                elif step.tool_name == "get_pr_diff":
                    pr_num = data.get("pr_number", "unknown")
                    new_evidence.append(
                        Evidence(
                            source_type="pull_request_diff",
                            source_id=f"PR-{pr_num}-diff",
                            content=data
                        )
                    )
                elif step.tool_name == "search_issues":
                    for issue in data:
                        new_evidence.append(
                            Evidence(
                                source_type="issue",
                                source_id=f"Issue-{issue.get('number', 'unknown')}",
                                url=issue.get("html_url"),
                                content=issue
                            )
                        )
                elif step.tool_name == "read_repository_file":
                    path = data.get("file_path", "unknown")
                    new_evidence.append(
                        Evidence(
                            source_type="file",
                            source_id=path,
                            content=data
                        )
                    )
                else:
                    new_evidence.append(
                        Evidence(
                            source_type="unknown",
                            source_id=f"{step.tool_name}-{step.id}",
                            content={"data": data}
                        )
                    )

            tool_history.append(
                ToolCallRecord(
                    step_id=step.id,
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    result=content
                )
            )

    except Exception as e:
        # Record failure cleanly
        tool_history.append(
            ToolCallRecord(
                step_id=step.id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                result={"success": False, "error": str(e)}
            )
        )
        
    return {
        "tool_calls_history": tool_history,
        "retrieved_evidence": new_evidence,
        "current_step": current_step_index + 1
    }
