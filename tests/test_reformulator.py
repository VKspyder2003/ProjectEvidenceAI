import pytest
from src.agent.state import AgentState, ToolCallRecord, FailureType
from src.agent.reformulator import reformulator_node
from langchain_core.runnables import RunnableConfig

class MockLLM:
    def __init__(self, text_to_return):
        self.text_to_return = text_to_return
        self.invocations = []
        
    async def ainvoke(self, messages):
        self.invocations.append(messages)
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(self.text_to_return)

class MockDependencies:
    def __init__(self, text_to_return):
        self.llm = MockLLM(text_to_return)

@pytest.mark.asyncio
async def test_reformulator_uses_history():
    deps = MockDependencies("Mock hint")
    config = RunnableConfig(configurable={"dependencies": deps})
    
    # Simulate a history where search_issues succeeded
    tool_history = [
        ToolCallRecord(
            step_id=0,
            tool_name="search_issues",
            arguments={"repo_owner": "octocat", "repo_name": "Hello-World"},
            result={"issues": []},
            failure_type=FailureType.NONE
        )
    ]
    
    state = AgentState(
        query="test",
        plan=[],
        current_step=0,
        plan_version=1,
        tool_calls_history=tool_history,
        last_failure={
            "tool_name": "read_repository_file",
            "arguments": {"branch": "main", "repo_owner": "octocat", "repo_name": "Hello-World"},
            "error": "404 Not Found"
        },
        retrieved_evidence=[],
        budget_consumed=0,
        retry_count=0,
        correction_hints=[],
        failed_step_id=None,
        session_context=None,
        draft_response=None,
        output_validation_result=None,
        fatal_error=False,
        error=None
    )
    
    updates = await reformulator_node(state, config)
    
    # Check if history was passed to the LLM
    invocations = deps.llm.invocations
    assert len(invocations) > 0
    system_prompt = invocations[0][0].content
    
    # The prompt should contain the execution history showing success
    assert "search_issues" in system_prompt
    assert "Success: True" in system_prompt
    
    # It should have produced a hint and recorded the failed step
    assert "Mock hint" in updates["correction_hints"]
    assert updates["failed_step_id"] == 0
