from langgraph.graph import StateGraph, START, END
from typing import Literal

from .state import AgentState
from .planner import planner_node
from .executor import executor_node
from .synthesizer import synthesizer_node

def should_continue_execution(state: AgentState) -> Literal["executor", "synthesizer"]:
    """
    Evaluates whether the executor should run again based on the plan and current step.
    """
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    
    if current_step < len(plan):
        return "executor"
    return "synthesizer"

def build_graph():
    """
    Compiles and returns the Phase 2 agentic workflow graph.
    No global LLM or MCP Client instances are created here.
    """
    builder = StateGraph(AgentState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "planner")
    
    # Planner always routes initially to executor
    builder.add_edge("planner", "executor")
    
    # Executor conditionally routes back to itself or concludes at synthesizer
    builder.add_conditional_edges(
        "executor",
        should_continue_execution,
        {
            "executor": "executor",
            "synthesizer": "synthesizer"
        }
    )
    
    builder.add_edge("synthesizer", END)
    
    return builder.compile()
