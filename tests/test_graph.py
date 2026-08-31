import pytest
from unittest.mock import patch, AsyncMock
from src.agent.graph import build_graph, evaluate_execution
from src.agent.state import PlanStep, ToolCallRecord, FailureType

def test_evaluate_execution_success():
    # Multi-step, unfinished
    assert evaluate_execution({"current_step": 0, "plan": [PlanStep(id=1, tool_name="a", arguments={}, reason="b")]}) == "executor"
    # Completed
    assert evaluate_execution({"current_step": 1, "plan": [PlanStep(id=1, tool_name="a", arguments={}, reason="b")]}) == "evidence_budget"
    # Empty plan
    assert evaluate_execution({"current_step": 0, "plan": []}) == "evidence_budget"

def test_evaluate_execution_fatal():
    # Explicit fatal_error flag
    assert evaluate_execution({"fatal_error": True}) == "evidence_budget"
    
    # Fatal from tool call history
    assert evaluate_execution({
        "tool_calls_history": [
            ToolCallRecord(step_id=1, tool_name="a", arguments={}, result="", failure_type=FailureType.FATAL)
        ]
    }) == "evidence_budget"

def test_evaluate_execution_recoverable():
    # Retries remaining
    assert evaluate_execution({
        "retry_count": 0,
        "tool_calls_history": [
            ToolCallRecord(step_id=1, tool_name="a", arguments={}, result="", failure_type=FailureType.RECOVERABLE)
        ]
    }) == "reformulator"
    
    # Exhausted retries
    assert evaluate_execution({
        "retry_count": 3,
        "tool_calls_history": [
            ToolCallRecord(step_id=1, tool_name="a", arguments={}, result="", failure_type=FailureType.RECOVERABLE)
        ]
    }) == "evidence_budget"

@pytest.mark.asyncio
async def test_graph_routing_single_step():
    with patch("src.agent.graph.planner_node", new_callable=AsyncMock) as mock_planner, \
         patch("src.agent.graph.executor_node", new_callable=AsyncMock) as mock_executor, \
         patch("src.agent.graph.reformulator_node", new_callable=AsyncMock) as mock_reformulator, \
         patch("src.agent.graph.evidence_budget_node", new_callable=AsyncMock) as mock_budget, \
         patch("src.agent.graph.synthesizer_node", new_callable=AsyncMock) as mock_synthesizer:
        
        mock_planner.return_value = {"plan": [PlanStep(id=1, tool_name="test", arguments={}, reason="test")]}
        
        def executor_side_effect(state, *args, **kwargs):
            step = state.get("current_step", 0)
            plan = state.get("plan", [])
            if step >= len(plan):
                return {}
            return {"current_step": step + 1}
            
        mock_executor.side_effect = executor_side_effect
        mock_budget.return_value = {}
        mock_synthesizer.return_value = {"draft_response": "done"}
        
        graph = build_graph()
        initial_state = {"query": "test", "current_step": 0}
        
        result = await graph.ainvoke(initial_state)
        
        assert mock_planner.call_count == 1
        assert mock_executor.call_count == 1
        assert mock_budget.call_count == 1
        assert mock_synthesizer.call_count == 1
        assert result["current_step"] == 1

@pytest.mark.asyncio
async def test_graph_routing_multi_step():
    with patch("src.agent.graph.planner_node", new_callable=AsyncMock) as mock_planner, \
         patch("src.agent.graph.executor_node", new_callable=AsyncMock) as mock_executor, \
         patch("src.agent.graph.reformulator_node", new_callable=AsyncMock) as mock_reformulator, \
         patch("src.agent.graph.evidence_budget_node", new_callable=AsyncMock) as mock_budget, \
         patch("src.agent.graph.synthesizer_node", new_callable=AsyncMock) as mock_synthesizer:
        
        mock_planner.return_value = {"plan": [
            PlanStep(id=1, tool_name="test1", arguments={}, reason="test"),
            PlanStep(id=2, tool_name="test2", arguments={}, reason="test")
        ]}
        
        def executor_side_effect(state, *args, **kwargs):
            step = state.get("current_step", 0)
            plan = state.get("plan", [])
            if step >= len(plan):
                return {}
            return {"current_step": step + 1}
            
        mock_executor.side_effect = executor_side_effect
        mock_budget.return_value = {}
        mock_synthesizer.return_value = {"draft_response": "done"}
        
        graph = build_graph()
        initial_state = {"query": "test", "current_step": 0}
        
        result = await graph.ainvoke(initial_state)
        
        assert mock_planner.call_count == 1
        assert mock_executor.call_count == 2
        assert mock_budget.call_count == 1
        assert mock_synthesizer.call_count == 1
        assert result["current_step"] == 2
