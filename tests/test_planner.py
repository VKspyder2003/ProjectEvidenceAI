import pytest
from src.agent.state import AgentState, PlanStep, ToolCallRecord, FailureType, SessionContext
from src.agent.planner import planner_node
from langchain_core.runnables import RunnableConfig

class MockStructuredLLM:
    def __init__(self, plan_to_return):
        self.plan_to_return = plan_to_return
        self.invocations = []
        
    async def ainvoke(self, messages):
        self.invocations.append(messages)
        return self.plan_to_return

class MockLLM:
    def __init__(self, plan_to_return):
        self.plan_to_return = plan_to_return
        
    def with_structured_output(self, schema):
        return MockStructuredLLM(self.plan_to_return)

class MockTool:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

class MockClient:
    def __init__(self, *args, **kwargs):
        self.tools = [MockTool("test_tool", "desc", {})]
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        pass
        
    async def list_tools(self):
        return self.tools

class MockDependencies:
    def __init__(self, plan_to_return):
        self.llm = MockLLM(plan_to_return)
        self.mcp_server = "mock_server"

@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setattr("src.agent.planner.Client", MockClient)

@pytest.mark.asyncio
async def test_planner_scoped_replanning(mock_client):
    from src.agent.state import ExecutionPlan
    
    # Simulate an LLM that returns both the failed step and the corrected step
    # to prove the exact-match filter works.
    llm_plan = ExecutionPlan(
        steps=[
            PlanStep(id=0, tool_name="test_tool", arguments={"branch": "main"}, reason="hallucinated duplicate of failure"),
            PlanStep(id=1, tool_name="test_tool", arguments={"branch": "master"}, reason="corrected step")
        ]
    )
    
    deps = MockDependencies(llm_plan)
    config = RunnableConfig(configurable={"dependencies": deps})
    
    old_plan = [
        PlanStep(id=0, tool_name="test_tool", arguments={"branch": "main"}, reason="initial failure step"),
        PlanStep(id=1, tool_name="test_tool", arguments={"other": "args"}, reason="unexecuted subsequent step")
    ]
    
    tool_history = [
        ToolCallRecord(
            step_id=0,
            tool_name="test_tool",
            arguments={"branch": "main"},
            result={"success": False, "error": "failed"},
            failure_type=FailureType.RECOVERABLE
        )
    ]
    
    state = AgentState(
        query="test",
        plan=old_plan,
        current_step=0,
        plan_version=1,
        tool_calls_history=tool_history,
        last_failure={"tool_name": "test_tool", "arguments": {"branch": "main"}},
        retrieved_evidence=[],
        budget_consumed=0,
        retry_count=1,
        correction_hints=["Try master"],
        failed_step_id=0,
        session_context=SessionContext(),
        draft_response=None,
        output_validation_result=None,
        fatal_error=False,
        error=None
    )
    
    updates = await planner_node(state, config)
    
    # Valid steps should have dropped the exact failure step
    assert "plan" in updates
    new_plan = updates["plan"]
    
    # Should only contain the corrected step
    assert len(new_plan) == 1
    assert new_plan[0].arguments == {"branch": "master"}
    assert new_plan[0].id == 0 # IDs are re-assigned sequentially
    assert updates["plan_version"] == 2
