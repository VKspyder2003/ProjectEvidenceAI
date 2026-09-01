import pytest
import sqlite3
import anyio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.agent.graph import build_graph
from src.agent.dependencies import AgentDependencies
from src.agent.state import SessionContext
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
