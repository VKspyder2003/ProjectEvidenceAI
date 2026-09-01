import json
import pytest
import asyncio
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict

from src.guardrails.validators import validate_input
from src.guardrails.output_validator import validate_output
from src.agent.graph import evaluate_execution
from src.agent.evidence_budget import evidence_budget_node
from src.agent.state import Evidence, PlanStep, ToolCallRecord, FailureType

class EvalResult(BaseModel):
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    failures_by_category: Dict[str, int] = {}
    
    def display(self):
        print("\n" + "="*50)
        print(" PHASE 4C.1 EVALUATION BENCHMARK RESULTS")
        print("="*50)
        print(f"Total cases:  {self.total_cases}")
        print(f"Passed cases: {self.passed_cases}")
        print(f"Failed cases: {self.failed_cases}")
        print(f"Pass Rate:    {self.pass_rate:.1f}%")
        
        if self.failed_cases > 0:
            print("\nFailures by Category:")
            for cat, count in self.failures_by_category.items():
                print(f"  - {cat}: {count}")
        print("="*50 + "\n")

def load_benchmark() -> List[dict]:
    p = Path(__file__).parent / "eval_benchmark.json"
    with open(p, "r") as f:
        return json.load(f)

@pytest.mark.asyncio
async def test_run_evaluation_benchmark(capsys):
    cases = load_benchmark()
    result = EvalResult()
    result.total_cases = len(cases)
    
    for case in cases:
        cat = case["category"]
        if cat not in result.failures_by_category:
            result.failures_by_category[cat] = 0
            
        passed = False
        
        try:
            if case["type"] == "input":
                res = validate_input(case["query"])
                passed = (res.is_allowed == case["expected_valid"])
                
            elif case["type"] == "output":
                evidence_objs = [
                    Evidence(source_type="test", source_id=e["id"], url=e.get("url"), content={"data": "test"})
                    for e in case.get("evidence", [])
                ]
                res = validate_output(case["response"], evidence_objs)
                passed = (res.valid == case["expected_valid"])
                
            elif case["type"] == "routing":
                state_data = case["state"]
                state = {
                    "current_step": state_data["current_step"],
                    "plan": [PlanStep(id=i, tool_name="x", arguments={}, reason="y") for i in range(state_data["plan_length"])],
                    "fatal_error": state_data["fatal_error"],
                    "retry_count": state_data["retry_count"]
                }
                if state_data["failure_type"] != "none":
                    ft_map = {
                        "recoverable": FailureType.RECOVERABLE,
                        "fatal": FailureType.FATAL,
                        "transient": FailureType.TRANSIENT
                    }
                    state["tool_calls_history"] = [
                        ToolCallRecord(step_id=1, tool_name="t", arguments={}, result="", failure_type=ft_map[state_data["failure_type"]])
                    ]
                
                route = evaluate_execution(state)
                passed = (route == case["expected_route"])
                
            elif case["type"] == "budget":
                evidence_objs = []
                for e in case["evidence"]:
                    content = "x" * (e["tokens"] * 4) # dummy content simulating length
                    evidence_objs.append(
                        Evidence(source_type="test", source_id=e["id"], url="url", content={"content": content}, token_estimate=e["tokens"])
                    )
                
                state = {"retrieved_evidence": evidence_objs}
                updates = await evidence_budget_node(state, {"configurable": {}})
                new_ev = updates["budgeted_evidence"]
                
                passed_count = len(new_ev) == case["expected_count"]
                
                passed_truncate = True
                if case.get("expect_truncate"):
                    passed_truncate = any("[TRUNCATED BY BUDGET]" in str(ev.content) for ev in new_ev)
                
                passed = passed_count and passed_truncate

        except Exception as e:
            print(f"Exception in case {case['id']}: {e}")
            passed = False
            
        if passed:
            result.passed_cases += 1
        else:
            result.failed_cases += 1
            result.failures_by_category[cat] += 1
            
    # Calculate pass rate
    result.pass_rate = (result.passed_cases / result.total_cases) * 100
    
    # Print the evaluation report
    with capsys.disabled():
        result.display()
        
    assert result.failed_cases == 0, f"Benchmark failed with {result.failed_cases} failures."
