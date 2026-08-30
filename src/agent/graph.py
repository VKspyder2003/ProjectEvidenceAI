from langgraph.graph import StateGraph, START, END
from typing import Literal

from .state import AgentState
from .planner import planner_node
from .executor import executor_node
from .synthesizer import synthesizer_node
from .reformulator import reformulator_node
from .evidence_budget import evidence_budget_node

def evaluate_execution(state: AgentState) -> Literal["executor", "evidence_budget", "reformulator"]:
    """
    Evaluates whether the executor should run again, recover, or finish.
    """
    if state.get("fatal_error"):
        return "evidence_budget"

    tool_history = state.get("tool_calls_history", [])
    if tool_history:
        latest = tool_history[-1]
        if latest.failure_type == "fatal":
            return "evidence_budget"
        elif latest.failure_type in ("recoverable", "transient"):
            retry_count = state.get("retry_count", 0)
            if retry_count < 3:
                return "reformulator"
            else:
                return "evidence_budget"

    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    
    if current_step < len(plan):
        return "executor"
    return "evidence_budget"

def build_graph():
    """
    Compiles and returns the Phase 2 agentic workflow graph.
    No global LLM or MCP Client instances are created here.
    """
    builder = StateGraph(AgentState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reformulator", reformulator_node)
    builder.add_node("evidence_budget", evidence_budget_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "planner")
    
    # Planner always routes initially to executor
    builder.add_edge("planner", "executor")
    
    # Executor conditionally routes
    builder.add_conditional_edges(
        "executor",
        evaluate_execution,
        {
            "executor": "executor",
            "evidence_budget": "evidence_budget",
            "reformulator": "reformulator"
        }
    )
    
    # Reformulator feeds back to planner to try again
    builder.add_edge("reformulator", "planner")
    
    # Evidence budget feeds to synthesizer
    builder.add_edge("evidence_budget", "synthesizer")
    
    builder.add_edge("synthesizer", END)
    
    return builder.compile()
