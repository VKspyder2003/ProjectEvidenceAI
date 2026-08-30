import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.agent.executor import executor_node
from src.agent.state import PlanStep
from src.agent.dependencies import AgentDependencies

@pytest.fixture
def mock_dependencies():
    return AgentDependencies(llm=MagicMock(), mcp_server=MagicMock())

@pytest.fixture
def mock_config(mock_dependencies):
    return {"configurable": {"dependencies": mock_dependencies}}

@pytest.mark.asyncio
async def test_executor_node_success(mock_config):
    state = {
        "plan": [
            PlanStep(
                id=1, 
                tool_name="get_recent_pull_requests", 
                arguments={"repo_owner": "test", "repo_name": "test"}, 
                reason="Test reason"
            )
        ],
        "current_step": 0
    }
    
    with patch("src.agent.executor.Client") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        
        mock_result = MagicMock()
        mock_result.structured_content = {
            "success": True,
            "data": [{"number": 42, "html_url": "https://github.com/test/test/pull/42"}],
            "error": None
        }
        mock_client_instance.call_tool.return_value = mock_result
        MockClient.return_value = mock_client_instance
        
        updates = await executor_node(state, mock_config)
        
        # Verify tool history
        assert "tool_calls_history" in updates
        assert len(updates["tool_calls_history"]) == 1
        assert updates["tool_calls_history"][0].result["success"] is True
        
        # Verify evidence transformation
        assert "retrieved_evidence" in updates
        assert len(updates["retrieved_evidence"]) == 1
        assert updates["retrieved_evidence"][0].source_type == "pull_request"
        assert updates["retrieved_evidence"][0].source_id == "PR-42"
        assert updates["retrieved_evidence"][0].url == "https://github.com/test/test/pull/42"
        
        # Verify step increment
        assert updates["current_step"] == 1

@pytest.mark.asyncio
async def test_executor_node_exception(mock_config):
    state = {
        "plan": [
            PlanStep(
                id=2, 
                tool_name="search_issues", 
                arguments={"repo_owner": "test", "repo_name": "test", "query": "bug"}, 
                reason="Find bugs"
            )
        ],
        "current_step": 0
    }
    
    with patch("src.agent.executor.Client") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        
        # Simulate network or client error
        mock_client_instance.call_tool.side_effect = Exception("MCP Connection Refused")
        MockClient.return_value = mock_client_instance
        
        updates = await executor_node(state, mock_config)
        
        # Verify failed tool history is preserved
        assert "tool_calls_history" in updates
        assert len(updates["tool_calls_history"]) == 1
        assert updates["tool_calls_history"][0].result["success"] is False
        assert "MCP Connection Refused" in updates["tool_calls_history"][0].result["error"]
        
        # Evidence should be empty for a failed call
        assert "retrieved_evidence" in updates
        assert len(updates["retrieved_evidence"]) == 0
        
        assert updates["current_step"] == 1

@pytest.mark.asyncio
async def test_executor_node_no_remaining_steps(mock_config):
    state = {
        "plan": [
            PlanStep(
                id=1, 
                tool_name="get_recent_pull_requests", 
                arguments={}, 
                reason="Done"
            )
        ],
        "current_step": 1 # Already completed
    }
    
    updates = await executor_node(state, mock_config)
    assert updates == {}
