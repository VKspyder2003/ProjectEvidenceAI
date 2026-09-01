import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage

from src.agent.synthesizer import synthesizer_node, format_evidence_for_prompt
from src.agent.state import Evidence
from src.agent.dependencies import AgentDependencies

@pytest.fixture
def mock_config():
    # Provide a mock LLM inside dependencies
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="This is the synthesized answer using [PR-42](https://github.com/owner/repo/pull/42).")
    
    deps = AgentDependencies(llm=mock_llm, mcp_server=MagicMock())
    return {"configurable": {"dependencies": deps}}

def test_format_evidence_helper():
    evidence = [
        Evidence(
            source_type="pull_request",
            source_id="PR-42",
            url="https://github.com/owner/repo/pull/42",
            content={"title": "Fix bug", "state": "merged"}
        ),
        Evidence(
            source_type="issue",
            source_id="Issue-57",
            url="https://github.com/owner/repo/issues/57",
            content={"title": "Database error"}
        )
    ]
    
    formatted = format_evidence_for_prompt(evidence)
    
    assert "SOURCE 1" in formatted
    assert "Type: pull_request" in formatted
    assert "ID: PR-42" in formatted
    assert "URL: https://github.com/owner/repo/pull/42" in formatted
    assert "Fix bug" in formatted
    
    assert "---" in formatted
    
    assert "SOURCE 2" in formatted
    assert "Type: issue" in formatted
    assert "ID: Issue-57" in formatted
    assert "URL: https://github.com/owner/repo/issues/57" in formatted
    assert "Database error" in formatted

@pytest.mark.asyncio
async def test_synthesizer_success(mock_config):
    state = {
        "query": "What is the status of PR 42?",
        "budgeted_evidence": [
            Evidence(
                source_type="pull_request",
                source_id="PR-42",
                url="https://github.com/owner/repo/pull/42",
                content={"title": "Fix bug", "state": "merged"}
            )
        ]
    }
    
    updates = await synthesizer_node(state, mock_config)
    
    assert updates["draft_response"] == "This is the synthesized answer using [PR-42](https://github.com/owner/repo/pull/42)."
    assert updates["error"] is None
    
    # Verify ainvoke was called
    llm = mock_config["configurable"]["dependencies"].llm
    llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_synthesizer_no_evidence(mock_config):
    state = {
        "query": "What is the status of PR 42?",
        "budgeted_evidence": []
    }
    
    updates = await synthesizer_node(state, mock_config)
    
    # Verify short-circuit
    assert "I couldn't find sufficient repository evidence" in updates["draft_response"]
    assert updates["error"] is None
    
    # LLM should not be called
    llm = mock_config["configurable"]["dependencies"].llm
    llm.ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_synthesizer_llm_failure(mock_config):
    state = {
        "query": "What is the status of PR 42?",
        "budgeted_evidence": [
             Evidence(
                source_type="pull_request",
                source_id="PR-42",
                url="https://github.com/owner/repo/pull/42",
                content={"title": "Fix bug", "state": "merged"}
            )
        ]
    }
    
    # Force LLM exception
    llm = mock_config["configurable"]["dependencies"].llm
    llm.ainvoke.side_effect = Exception("API rate limit exceeded")
    
    updates = await synthesizer_node(state, mock_config)
    
    # Verify graceful error handling
    assert updates["draft_response"] is None
    assert updates["error"] is not None
    assert "Synthesis failed: API rate limit exceeded" in updates["error"]
