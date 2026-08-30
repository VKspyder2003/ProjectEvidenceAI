import os
import asyncio
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

# Force environment config for the test
os.environ["LLM_PROVIDER"] = "groq"
os.environ["LLM_MODEL"] = "qwen/qwen3.8-27b"

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
        ("Test B", "Search for open issues related to documentation in octocat/Hello-World and summarize the results."),
        ("Test C", "Find recent pull requests and open issues related to documentation in octocat/Hello-World. Summarize the current state and identify anything that may need attention.")
    ]
    
    for test_name, query in queries:
        print(f"\n[{test_name}] Query: {query}")
        print("=" * 60)
        
        initial_state = {
            "query": query,
            "plan": [],
            "current_step": 0,
            "tool_calls_history": [],
            "retrieved_evidence": []
        }
        
        config = {
            "configurable": {
                "dependencies": deps
            }
        }
        
        print("Executing Graph Workflow...\n")
        try:
            async for event in graph.astream(initial_state, config=config):
                for node_name, state_updates in event.items():
                    print(f"--- Node Executed: {node_name.upper()} ---")
                    
                    state_updates = state_updates or {}
                    
                    if node_name == "planner":
                        err = state_updates.get("error")
                        if err:
                            print(f"PLANNER ERROR: {err}")
                        plan = state_updates.get("plan", [])
                        print(f"Generated {len(plan)} plan steps:")
                        for step in plan:
                            print(f"  [{step.id}] {step.tool_name}({step.arguments})")
                            print(f"      Reason: {step.reason}")
                            
                    elif node_name == "executor":
                        history = state_updates.get("tool_calls_history", [])
                        if history:
                            latest = history[-1]
                            success = latest.result.get("success") if isinstance(latest.result, dict) else "Unknown"
                            print(f"Executed step {latest.step_id} -> {latest.tool_name}")
                            print(f"Success: {success}")
                            if not success:
                                print(f"Error: {latest.result.get('error')}")
                                
                    elif node_name == "synthesizer":
                        err = state_updates.get("error")
                        if err:
                            print(f"\n=== SYNTHESIS ERROR ===\n{err}\n==========================")
                        else:
                            print("\n=== GROUNDED SYNTHESIS ===")
                            print(state_updates.get("draft_response"))
                            print("==========================")
                    print()
        except Exception as e:
            print(f"Graph execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_tests())
