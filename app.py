import os
import uuid
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Default to groq if not explicitly set
if "LLM_PROVIDER" not in os.environ:
    os.environ["LLM_PROVIDER"] = "groq"

from src.agent.dependencies import get_agent_dependencies
from src.agent.graph import build_graph
from src.agent.state import SessionContext
from src.guardrails.validators import validate_input

# --- 1. Initialization ---
st.set_page_config(page_title="ProjectEvidenceAI", page_icon="🔍", layout="centered")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "repo_input" not in st.session_state:
    st.session_state.repo_input = "octocat/Hello-World"

@st.cache_resource
def get_deps():
    return get_agent_dependencies()

deps = get_deps()

def new_session():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []

# --- 2. Layout ---
st.title("ProjectEvidenceAI")
st.markdown("Evidence-grounded agentic analysis over live GitHub data")

if len(st.session_state.chat_history) == 0:
    st.text_input("Repository (owner/repo)", key="repo_input", placeholder="e.g. octocat/Hello-World")
else:
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.caption(f"**Repository:** {st.session_state.get('repo_input', 'octocat/Hello-World')} | **Session ID:** {st.session_state.thread_id[:8]}")
    with col2:
        if st.button("New Session"):
            new_session()
            st.rerun()

# --- 3. Chat Interface ---
# Display previous chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Agent Trace"):
                for t in msg["trace"]:
                    st.markdown(t)
        if msg.get("metrics"):
            st.caption(msg["metrics"])
        if msg.get("citations"):
            st.markdown("**Verified Sources**")
            for cit in msg["citations"]:
                st.markdown(f"- [{cit.get('source_id', 'Unknown')}]({cit.get('url', '#')})")

if prompt := st.chat_input("What changed recently and what needs attention?"):
    # Add user message to state
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 4. Input Guardrail Verification
        is_valid, reason = validate_input(prompt)
        if not is_valid:
            error_msg = f"**Blocked by Input Guardrail:** {reason}"
            st.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.stop()

        initial_state = {
            "query": prompt,
        }
        
        # Only parse and inject session context on the first query of the session
        if len(st.session_state.chat_history) <= 1:
            repo_parts = st.session_state.get("repo_input", "").split("/")
            if len(repo_parts) != 2:
                st.error("Invalid repository format. Must be 'owner/repo'.")
                st.stop()
            repo_owner, repo_name = repo_parts
            initial_state["session_context"] = SessionContext(repo_owner=repo_owner, repo_name=repo_name)

        config = {
            "configurable": {
                "dependencies": deps,
                "thread_id": st.session_state.thread_id
            }
        }

        trace_log = ["✓ Input Guardrail passed"]
        metrics_str = ""
        citations = []
        final_answer = ""
        
        status = st.status("Agent is analyzing...", expanded=True)
        status.write("✓ Input Guardrail passed")
        
        async def run_agent():
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            
            async with AsyncSqliteSaver.from_conn_string("data/agent_memory.db") as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                
                async for event in graph.astream(initial_state, config=config):
                    for node_name, state_updates in event.items():
                        state_updates = state_updates or {}
                        
                        if node_name == "planner":
                            plan = state_updates.get("plan", [])
                            if plan:
                                msg = f"✓ Planner generated {len(plan)} steps"
                            else:
                                msg = "⚠ Planner generated 0 steps"
                            status.write(msg)
                            trace_log.append(msg)
                        elif node_name == "executor":
                            history = state_updates.get("tool_calls_history", [])
                            if history:
                                latest = history[-1]
                                success = latest.result.get("success") if isinstance(latest.result, dict) else False
                                if success:
                                    msg = f"✓ Executed `{latest.tool_name}` successfully"
                                else:
                                    failure_type = latest.failure_type.value if hasattr(latest.failure_type, 'value') else str(latest.failure_type)
                                    msg = f"⚠ Tool failure: `{latest.tool_name}` ({failure_type})"
                                status.write(msg)
                                trace_log.append(msg)
                        elif node_name == "reformulator":
                            hints = state_updates.get("correction_hints", [])
                            if hints:
                                msg = f"⚠ Reformulator injected correction hint"
                            else:
                                msg = "⚠ Reformulator executing"
                            status.write(msg)
                            trace_log.append(msg)
                        elif node_name == "evidence_budget":
                            evidence = state_updates.get("budgeted_evidence", [])
                            budget = sum(e.token_estimate for e in evidence) if evidence else 0
                            msg = f"✓ Evidence Budget: {len(evidence)} items (~{budget} tokens)"
                            status.write(msg)
                            trace_log.append(msg)
                        elif node_name == "synthesizer":
                            err = state_updates.get("error")
                            if err:
                                msg = "✖ Synthesis failed"
                                status.error(msg)
                            else:
                                msg = "✓ Synthesizer generated draft response"
                                status.write(msg)
                            trace_log.append(msg)
                        elif node_name == "output_validator":
                            msg = "✓ Citation Validator complete"
                            status.write(msg)
                            trace_log.append(msg)
                
                # Fetch final state using async method (LangGraph API)
                return await graph.aget_state(config)
            
        try:
            # Run graph execution in event loop
            final_snapshot = asyncio.run(run_agent())
            final_state = final_snapshot.values
            
            if final_state:
                err = final_state.get("error")
                plan = final_state.get("plan", [])
                
                if err:
                    status.update(label="Analysis failed due to an error.", state="error", expanded=True)
                elif len(plan) == 0:
                    status.update(label="No executable plan was required or generated.", state="complete", expanded=False)
                else:
                    status.update(label="Analysis complete", state="complete", expanded=False)
                    
                final_answer = final_state.get("draft_response", "No response generated.")
                if not final_answer:
                    err = final_state.get("error")
                    if err:
                        final_answer = f"**Error:** {err}"
                        
                # Extract citations
                validation_result = final_state.get("output_validation_result")
                if validation_result:
                    if hasattr(validation_result, "model_dump"):
                        vr = validation_result.model_dump()
                    else:
                        vr = validation_result
                    if vr.get("validated_citations"):
                        citations = [c for c in vr["validated_citations"] if c.get("is_valid")]
                        
                # Extract metrics
                history = final_state.get("tool_calls_history", [])
                total_tools = len(history)
                failed_tools = sum(1 for h in history if not (h.result.get("success") if isinstance(h.result, dict) else False))
                retries = final_state.get("retry_count", 0)
                evidence = final_state.get("retrieved_evidence", [])
                est_tokens = sum(e.token_estimate for e in evidence) if evidence else 0
                
                metrics_str = f"Metrics: {total_tools} tool calls ({failed_tools} failed) | {retries} retries | {len(evidence)} evidence items (~{est_tokens} tokens)"
                
            st.markdown(final_answer)
            if citations:
                st.markdown("**Verified Sources**")
                for cit in citations:
                    st.markdown(f"- [{cit.get('source_id', 'Unknown')}]({cit.get('url', '#')})")
            
            if metrics_str:
                st.caption(metrics_str)
                
            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer,
                "trace": trace_log,
                "metrics": metrics_str,
                "citations": citations
            })
            
        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error("An internal error occurred during execution.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Fatal Error:** {str(e)}",
                "trace": trace_log
            })
