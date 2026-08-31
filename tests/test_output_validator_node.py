import pytest
from unittest.mock import patch, MagicMock
from src.agent.output_validator_node import output_validator_node
from src.agent.state import Evidence

@pytest.fixture
def mock_config():
    return {"configurable": {"dependencies": MagicMock()}}

@pytest.mark.asyncio
async def test_output_validator_node_valid_response(mock_config):
    state = {
        "draft_response": "Fixed in [PR-11049](https://github.com/octocat/Hello-World/pull/11049).",
        "retrieved_evidence": [
            Evidence(
                source_type="pull_request",
                source_id="PR-11049",
                url="https://github.com/octocat/Hello-World/pull/11049",
                content={"title": "Fix something"}
            )
        ]
    }
    
    updates = await output_validator_node(state, mock_config)
    
    # Assert validation result is stored
    assert "output_validation_result" in updates
    assert updates["output_validation_result"]["valid"] is True
    
    # Assert draft_response is NOT overwritten
    assert "draft_response" not in updates

@pytest.mark.asyncio
async def test_output_validator_node_invalid_citation(mock_config):
    state = {
        "draft_response": "Fixed in [Fabricated](https://github.com/octocat/Hello-World/pull/9999).",
        "retrieved_evidence": [
            Evidence(
                source_type="pull_request",
                source_id="PR-11049",
                url="https://github.com/octocat/Hello-World/pull/11049",
                content={"title": "Fix something"}
            )
        ]
    }
    
    updates = await output_validator_node(state, mock_config)
    
    # Assert validation result is stored
    assert "output_validation_result" in updates
    assert updates["output_validation_result"]["valid"] is False
    assert len(updates["output_validation_result"]["violations"]) == 1
    
    # Assert draft_response IS replaced with fallback
    assert "draft_response" in updates
    assert "could not be fully verified" in updates["draft_response"]

@pytest.mark.asyncio
async def test_output_validator_node_multiple_invalid_citations(mock_config):
    state = {
        "draft_response": "Fixed in [Fabricated](https://github.com/octocat/Hello-World/pull/9999) and [Fabricated2](https://github.com/octocat/Hello-World/pull/8888).",
        "retrieved_evidence": [
            Evidence(
                source_type="pull_request",
                source_id="PR-11049",
                url="https://github.com/octocat/Hello-World/pull/11049",
                content={"title": "Fix something"}
            )
        ]
    }
    
    updates = await output_validator_node(state, mock_config)
    
    # Assert validation result is stored
    assert "output_validation_result" in updates
    assert updates["output_validation_result"]["valid"] is False
    # Both violations should be preserved
    assert len(updates["output_validation_result"]["violations"]) == 2
    
    # Assert draft_response IS replaced with fallback
    assert "draft_response" in updates
    assert "could not be fully verified" in updates["draft_response"]
