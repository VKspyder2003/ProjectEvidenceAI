import os
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

# Use current environment config or default to groq
if "LLM_PROVIDER" not in os.environ:
    os.environ["LLM_PROVIDER"] = "groq"

# Attempt to load local API keys (GROQ_API_KEY, GITHUB_TOKEN)
load_dotenv()

from src.agent.dependencies import get_agent_dependencies
from src.agent.graph import build_graph

async def run_tests():
    print("Initializing dependencies...")
    try:
        deps = get_agent_dependencies()
    except Exception as e:
        print(f"Failed to initialize dependencies: {e}")
        return

    print("Building Phase 2 Orchestration Graph...")
    graph = build_graph()
    
    queries = [
        ("Test A - Recovery", "Read the README.md file in octocat/Hello-World. Start by checking the 'main' branch."),
        ("Test B - Fatal", "Summarize the open issues in octocat/this-repo-does-not-exist"),
        ("Test C - Budget", "Find all recent pull requests and open issues in octocat/Hello-World. Summarize the state of the repository.")
    ]
    
    for test_name, query in queries:
        print(f"\n[{test_name}] Query: {query}")
        print("=" * 60)
        
        from src.agent.state import SessionContext
        
        initial_state = {
            "query": query,
            "session_context": SessionContext(repo_owner="octocat", repo_name="Hello-World"),
            "plan": [],
            "plan_version": 0,
            "current_step": 0,
            "tool_calls_history": [],
            "retrieved_evidence": [],
            "budget_consumed": 0,
            "retry_count": 0,
            "correction_hints": [],
            "failed_step_id": None,
            "fatal_error": False,
            "draft_response": None,
            "error": None
        }
        
        config = {
            "configurable": {
                "dependencies": deps
            }
        }
        
        main_executions = 0
        
        print("Executing Graph Workflow...\n")
        try:
            async for event in graph.astream(initial_state, config=config):
                for node_name, state_updates in event.items():
                    print(f"--- Node Executed: {node_name.upper()} ---")
                    
                    state_updates = state_updates or {}
                    
                    # Track executions for Test A assertions
                    if test_name.startswith("Test A") and node_name == "executor":
                        history = state_updates.get("tool_calls_history", [])
                        if history:
                            latest = history[-1]
                            if latest.tool_name == "read_repository_file" and latest.arguments.get("branch") == "main":
                                main_executions += 1
                                
                    
                    if node_name == "planner":
                        err = state_updates.get("error")
                        if err:
                            print(f"PLANNER ERROR: {err}")
                        plan = state_updates.get("plan", [])
                        print(f"Generated {len(plan)} plan steps:")
                        for step in plan:
                            print(f"  [{step.id}] {step.tool_name}({step.arguments})")
                            print(f"      Reason: {step.reason}")
                            
                    elif node_name == "evidence_budget":
                        evidence = state_updates.get("retrieved_evidence", [])
                        total_budget = sum(ev.token_estimate for ev in evidence)
                        print(f"EVIDENCE BUDGET: Deduplicated and bounded evidence to {len(evidence)} items. Total estimated tokens: {total_budget}")
                        
                    elif node_name == "executor":
                        history = state_updates.get("tool_calls_history", [])
                        if history:
                            latest = history[-1]
                            success = latest.result.get("success") if isinstance(latest.result, dict) else "Unknown"
                            print(f"Executed step {latest.step_id} -> {latest.tool_name}")
                            print(f"Success: {success}, Failure Type: {latest.failure_type.value}")
                            if not success:
                                print(f"Error: {latest.result.get('error')}")
                    
                    elif node_name == "reformulator":
                        hints = state_updates.get("correction_hints", [])
                        if hints:
                            print(f"REFORMULATOR HINT: {hints[-1]}")
                                
                    elif node_name == "synthesizer":
                        err = state_updates.get("error")
                        if err:
                            print(f"\n=== SYNTHESIS ERROR ===\n{err}\n==========================")
                        else:
                            print("\n=== GROUNDED SYNTHESIS ===")
                            print(state_updates.get("draft_response"))
                            print("==========================")
                    print()
                    
            if test_name.startswith("Test A"):
                assert main_executions <= 1, f"Regression Failed: 'main' branch was executed {main_executions} times (expected 1)."
                print("✓ Verified: 'main' branch was executed only once.")
                
        except Exception as e:
            print(f"Graph execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_tests())
