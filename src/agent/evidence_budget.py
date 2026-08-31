import json
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from .state import AgentState, Evidence

# To safely fit within an 8000 TPM limit (e.g. for Groq's free tier Qwen model),
# we budget input tokens to 5000.
MAX_SYNTHESIS_INPUT_TOKENS = 5000
SYSTEM_PROMPT_OVERHEAD = 1000
OUTPUT_TOKEN_RESERVATION = 2000
MAX_ESTIMATED_EVIDENCE_TOKENS = MAX_SYNTHESIS_INPUT_TOKENS - SYSTEM_PROMPT_OVERHEAD - OUTPUT_TOKEN_RESERVATION

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
    # 3. Enforce token limit robustly
    final_evidence = []
    total_tokens = 0
    
    for ev in deduped:
        # Pre-serialize to capture the actual size of the entire object (including hidden huge fields)
        if not isinstance(ev.content, str):
            try:
                content_str = json.dumps(ev.content)
            except Exception:
                content_str = str(ev.content)
        else:
            content_str = ev.content
            
        actual_tokens = len(content_str) // 4
        
        if total_tokens + actual_tokens > MAX_ESTIMATED_EVIDENCE_TOKENS:
            # We must truncate or drop
            remaining_budget = MAX_ESTIMATED_EVIDENCE_TOKENS - total_tokens
            if remaining_budget > 100:
                char_limit = remaining_budget * 4
                truncated = content_str[:char_limit] + "\n...[TRUNCATED BY BUDGET]"
                ev.content = truncated
                ev.token_estimate = remaining_budget
                total_tokens += remaining_budget
                final_evidence.append(ev)
            break # Stop adding evidence
        else:
            ev.content = content_str
            ev.token_estimate = actual_tokens
            total_tokens += actual_tokens
            final_evidence.append(ev)
            
    # Return the bounded evidence list
    return {
        "retrieved_evidence": final_evidence
    }
