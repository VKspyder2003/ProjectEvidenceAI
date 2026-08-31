import os
import asyncio
import sys
import uuid
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

# Ensure we have required directories
os.makedirs("data", exist_ok=True)

# Force environment config for the test
os.environ["LLM_PROVIDER"] = "groq"
os.environ["LLM_MODEL"] = "qwen/qwen3.8-27b"

# Attempt to load local API keys (GROQ_API_KEY, GITHUB_TOKEN)
load_dotenv()

from src.agent.dependencies import get_agent_dependencies
from src.agent.graph import build_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def run_recovery_demo():
    print("=" * 60)
    print("PROJECTEVIDENCEAI — SELF-RECOVERY DEMO")
    print("=" * 60)

    try:
        deps = get_agent_dependencies()
    except Exception as e:
        print(f"Failed to initialize dependencies: {e}")
        return

    # Use the AsyncSqliteSaver checkpointer context
    async with AsyncSqliteSaver.from_conn_string("data/agent_memory.db") as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        
        # We explicitly request the main branch to guarantee a 404 on this legacy repo
        query = "Read the 'README' file (exactly that name, no .md extension) in octocat/Hello-World. Try the 'main' branch first. Include the citation URL exactly as provided in the evidence."
        print(f"\nUSER QUERY:\n{query}\n")
        
        initial_state = {
            "query": query,
            # Start with fresh state
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
        
        # Give this a unique thread_id for persistent state isolation
        demo_thread_id = f"recovery-demo-{uuid.uuid4()}"
        config = {
            "configurable": {
                "thread_id": demo_thread_id,
                "dependencies": deps
            }
        }
        
        try:
            async for event in graph.astream(initial_state, config=config):
                for node_name, state_updates in event.items():
                    state_updates = state_updates or {}
                    
                    if node_name == "planner":
                        err = state_updates.get("error")
                        if err:
                            print(f"PLANNER ERROR:\n{err}")
                        plan = state_updates.get("plan", [])
                        if plan:
                            print(f"PLAN (Version {state_updates.get('plan_version', 1)}):")
                            for step in plan:
                                print(f"  {step.tool_name}({step.arguments})")
                            print()
                            
                    elif node_name == "executor":
                        history = state_updates.get("tool_calls_history", [])
                        if history:
                            latest = history[-1]
                            success = latest.result.get("success") if isinstance(latest.result, dict) else "Unknown"
                            print("EXECUTOR:")
                            if success:
                                print("  ✓ SUCCESS")
                            else:
                                print("  ✗ FAILED")
                                print(f"  FailureType: {latest.failure_type.value.upper()}")
                                err_msg = latest.result.get('error', 'Unknown Error')
                                print(f"  Reason: {err_msg}")
                                # Look for HTTP status in the error string
                                if "404" in err_msg:
                                    print("  HTTP: 404")
                            print()
                    
                    elif node_name == "reformulator":
                        hints = state_updates.get("correction_hints", [])
                        if hints:
                            print("REFORMULATOR:")
                            print(f"  → Correction hint: {hints[-1]}\n")
                                
                    elif node_name == "evidence_budget":
                        evidence = state_updates.get("retrieved_evidence", [])
                        total_budget = sum(ev.token_estimate for ev in evidence)
                        print("EVIDENCE BUDGET:")
                        print(f"  Sources: {len(evidence)}")
                        print(f"  Estimated tokens: {total_budget}\n")
                                
                    elif node_name == "synthesizer":
                        err = state_updates.get("error")
                        if err:
                            print(f"SYNTHESIZER ERROR:\n{err}\n")
                        else:
                            print("SYNTHESIZER:")
                            print("  ✓ Generated grounded response\n")
                            
                    elif node_name == "output_validator":
                        val_result_dict = state_updates.get("output_validation_result")
                        if val_result_dict:
                            print("OUTPUT VALIDATOR:")
                            if val_result_dict.get("valid", False):
                                print("  ✓ Citations verified\n")
                            else:
                                print("  ✗ Citation validation failed")
                                print(f"  Reason: {val_result_dict.get('reason', '')}\n")
                                
            # Final output is retrieved directly from the final state structure
            final_state = (await graph.aget_state(config)).values
            print("============================================================")
            print("FINAL ANSWER:")
            print("============================================================")
            print(final_state.get("draft_response"))
            print("============================================================")
            
        except Exception as e:
            print(f"Graph execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_recovery_demo())
