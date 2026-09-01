import pytest
import sqlite3
import anyio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.agent.graph import build_graph
from src.agent.dependencies import AgentDependencies
from src.agent.state import SessionContext, ToolCallRecord, Evidence, PlanStep
from unittest.mock import MagicMock
from langchain_core.language_models import BaseChatModel

@pytest.mark.asyncio
async def test_graph_persistence():
    """
    Tests that the LangGraph SQLite checkpointer successfully isolates 
    thread state and persists state between invocations, matching
    the async invocation pattern used in the Streamlit app.
    """
    # Use an in-memory SQLite database for the test using the async saver
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        
        # Use deterministic dummy dependencies to ensure CI doesn't need API keys
        deps = AgentDependencies(
            llm=MagicMock(spec=BaseChatModel),
            mcp_server=MagicMock()
        )
        
        # 1. Run Session A
        config_a = {"configurable": {"thread_id": "session-a", "dependencies": deps}}
        
        # Fulfilling the Pydantic contract
        context_a = SessionContext(repo_owner="test", repo_name="repo-a")
        await graph.aupdate_state(config_a, {"query": "Session A Query", "session_context": context_a}, as_node="planner")
        
        # 2. Run Session B
        config_b = {"configurable": {"thread_id": "session-b", "dependencies": deps}}
        context_b = SessionContext(repo_owner="test", repo_name="repo-b")
        await graph.aupdate_state(config_b, {"query": "Session B Query", "session_context": context_b}, as_node="planner")
        
        # 3. Verify Isolation
        state_a = (await graph.aget_state(config_a)).values
        state_b = (await graph.aget_state(config_b)).values
        
        assert state_a["query"] == "Session A Query"
        assert state_a["session_context"] == context_a
        assert state_a["session_context"].repo_name == "repo-a"
        
        assert state_b["query"] == "Session B Query"
        assert state_b["session_context"] == context_b
        assert state_b["session_context"].repo_name == "repo-b"
        
        # 4. Update Session A and verify persistence
        updated_context_a = SessionContext(repo_owner="test", repo_name="repo-a-updated")
        await graph.aupdate_state(config_a, {"session_context": updated_context_a}, as_node="planner")
        
        state_a_updated = (await graph.aget_state(config_a)).values
        assert state_a_updated["session_context"] == updated_context_a
        assert state_a_updated["session_context"].repo_name == "repo-a-updated"
        assert state_a_updated["query"] == "Session A Query"  # Unchanged field remains

@pytest.mark.asyncio
async def test_multi_turn_isolation():
    """
    Tests that a new turn for the same thread correctly clears the old 
    execution state (plan, evidence) while retaining session context,
    as specified by the 'add_or_clear' reducer pattern.
    """
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        deps = AgentDependencies(
            llm=MagicMock(spec=BaseChatModel),
            mcp_server=MagicMock()
        )
        config = {"configurable": {"thread_id": "multi-turn-session", "dependencies": deps}}
        
        # Turn 1: Python Query
        context = SessionContext(repo_owner="microsoft", repo_name="vscode")
        turn1_state = {
            "query": "Find open issues related to Python",
            "session_context": context,
            "plan": [],
            "current_step": 0,
            "tool_calls_history": "clear",
            "retrieved_evidence": "clear"
        }
        await graph.aupdate_state(config, turn1_state, as_node="planner")
        
        # Simulate some execution in Turn 1
        await graph.aupdate_state(
            config, 
            {
                "plan": [PlanStep(id=0, tool_name="search_issues", arguments={"query": "Python"}, reason="")],
                "current_step": 1,
                "tool_calls_history": [ToolCallRecord(step_id=0, tool_name="search", arguments={}, result={}, failure_type="none")],
                "retrieved_evidence": [Evidence(source_type="issue", source_id="1", content={})]
            },
            as_node="executor"
        )
        
        state_after_turn1 = (await graph.aget_state(config)).values
        assert len(state_after_turn1["plan"]) == 1
        assert len(state_after_turn1["retrieved_evidence"]) == 1
        assert state_after_turn1["session_context"].repo_name == "vscode"
        
        # Turn 2: JavaScript Query (simulating app.py initial_state)
        turn2_state = {
            "query": "Are there any issues open for JavaScript?",
            "plan": [],
            "current_step": 0,
            "tool_calls_history": "clear",
            "retrieved_evidence": "clear",
        }
        await graph.aupdate_state(config, turn2_state, as_node="planner")
        
        state_start_turn2 = (await graph.aget_state(config)).values
        
        # TEST 1: Query replacement
        assert state_start_turn2["query"] == "Are there any issues open for JavaScript?"
        
        # TEST 2 & 3: Plan and Evidence isolation
        assert len(state_start_turn2["plan"]) == 0
        assert state_start_turn2["current_step"] == 0
        assert len(state_start_turn2["tool_calls_history"]) == 0
        assert len(state_start_turn2["retrieved_evidence"]) == 0
        
        # TEST 4: Repository context persistence
        assert state_start_turn2["session_context"].repo_owner == "microsoft"
        assert state_start_turn2["session_context"].repo_name == "vscode"
