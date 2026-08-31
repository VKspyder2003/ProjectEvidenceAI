from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from .state import AgentState, Evidence

MAX_SYNTHESIS_INPUT_TOKENS = 1500
SYSTEM_PROMPT_OVERHEAD = 500
MAX_ESTIMATED_EVIDENCE_TOKENS = MAX_SYNTHESIS_INPUT_TOKENS - SYSTEM_PROMPT_OVERHEAD

async def evidence_budget_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Evidence budget node that normalizes, deduplicates, and limits evidence
    before it is passed to the synthesizer.
    """
    retrieved_evidence = state.get("retrieved_evidence", [])
    
    # 1. Deduplicate by source_id
    seen_ids = set()
    deduped = []
    for ev in retrieved_evidence:
        if ev.source_id not in seen_ids:
            seen_ids.add(ev.source_id)
            deduped.append(ev)
            
    # 2. Sort or prioritize (e.g., diffs and files before issues, but we'll just keep order for now)
    # 3. Enforce token limit
    final_evidence = []
    total_tokens = 0
    
    for ev in deduped:
        if total_tokens + ev.token_estimate > MAX_ESTIMATED_EVIDENCE_TOKENS:
            # We must truncate or drop
            remaining_budget = MAX_ESTIMATED_EVIDENCE_TOKENS - total_tokens
            if remaining_budget > 100:
                # Try to truncate content if it's a file or issue body
                # Roughly 4 chars per token
                char_limit = remaining_budget * 4
                
                new_content = dict(ev.content)
                if "content" in new_content and isinstance(new_content["content"], str):
                    new_content["content"] = new_content["content"][:char_limit] + "\n...[TRUNCATED BY BUDGET]"
                    ev.content = new_content
                    total_tokens += remaining_budget
                    final_evidence.append(ev)
                elif "body" in new_content and isinstance(new_content["body"], str):
                    new_content["body"] = new_content["body"][:char_limit] + "\n...[TRUNCATED BY BUDGET]"
                    ev.content = new_content
                    total_tokens += remaining_budget
                    final_evidence.append(ev)
            break # Stop adding evidence
        else:
            total_tokens += ev.token_estimate
            final_evidence.append(ev)
            
    # Return the bounded evidence list
    return {
        "retrieved_evidence": final_evidence
    }
