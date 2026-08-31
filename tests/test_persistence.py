import pytest
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agent.graph import build_graph
from src.agent.dependencies import get_agent_dependencies

def test_graph_persistence():
    """
    Tests that the LangGraph SQLite checkpointer successfully isolates 
    thread state and persists state between invocations.
    """
    # Use an in-memory SQLite database for the test
    with sqlite3.connect(":memory:", check_same_thread=False) as conn:
        checkpointer = SqliteSaver(conn)
        graph = build_graph(checkpointer=checkpointer)
        deps = get_agent_dependencies()
        
        # 1. Run Session A
        config_a = {"configurable": {"thread_id": "session-a", "dependencies": deps}}
        
        # We invoke the graph with a simple query.
        initial_state_a = {"query": "Summarize PR #1"}
        
        # In a unit test, we want to just step through or ensure it runs.
        # But we don't have mock tools configured globally here, so it might 
        # attempt an actual MCP call if it gets to executor. 
        # Actually, if we just want to test persistence, we can run a partial graph 
        # or use a mock dependency. Let's just use a fake state and check if it persists.
        # A simple way to test persistence without executing nodes that do network calls 
        # is to manually update the state using update_state.
        
        graph.update_state(config_a, {"query": "Session A Query", "session_context": "Context A"})
        
        # 2. Run Session B
        config_b = {"configurable": {"thread_id": "session-b", "dependencies": deps}}
        graph.update_state(config_b, {"query": "Session B Query", "session_context": "Context B"})
        
        # 3. Verify Isolation
        state_a = graph.get_state(config_a).values
        state_b = graph.get_state(config_b).values
        
        assert state_a["query"] == "Session A Query"
        assert state_a["session_context"] == "Context A"
        
        assert state_b["query"] == "Session B Query"
        assert state_b["session_context"] == "Context B"
        
        # 4. Update Session A and verify persistence
        graph.update_state(config_a, {"session_context": "Updated Context A"})
        state_a_updated = graph.get_state(config_a).values
        assert state_a_updated["session_context"] == "Updated Context A"
        assert state_a_updated["query"] == "Session A Query"  # Unchanged field remains
