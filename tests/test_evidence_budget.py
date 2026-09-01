import pytest
import json
import asyncio
from src.agent.evidence_budget import (
    evidence_budget_node, 
    MAX_SYNTHESIS_INPUT_TOKENS, 
    MAX_ESTIMATED_EVIDENCE_TOKENS
)
from src.agent.synthesizer import format_evidence_for_prompt
from src.agent.prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.agent.state import Evidence

@pytest.mark.asyncio
async def test_complete_synthesis_input_budget():
    """
    Proves that even with massive, deeply nested hidden fields in the evidence object,
    the evidence_budget_node restricts the COMPLETE synthesis input (system prompt + formatted evidence)
    to stay strictly under MAX_SYNTHESIS_INPUT_TOKENS.
    """
    # Create massive evidence that attempts to blow up the prompt
    huge_metadata = "x" * 20000  # 20k characters (~5000 tokens)
    
    evidence_list = [
        Evidence(
            source_type="issue",
            source_id="issue-1",
            url="http://github.com",
            content={"body": "short body", "hidden_metadata": huge_metadata},
            token_estimate=5  # Purposely wrong token_estimate to simulate mismatch
        ),
        Evidence(
            source_type="pull_request",
            source_id="pr-2",
            url="http://github.com",
            content={"diff": "short diff", "huge_commits": huge_metadata},
            token_estimate=5
        )
    ]
    
    state = {"retrieved_evidence": evidence_list}
    config = {"configurable": {}}
    
    # Run budget node
    updates = await evidence_budget_node(state, config)
    bounded_evidence = updates.get("budgeted_evidence", [])
    
    assert len(bounded_evidence) > 0, "Should retain at least some evidence"
    
    # Format evidence and create prompt exactly as synthesizer does
    formatted = format_evidence_for_prompt(bounded_evidence)
    system_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(evidence_context=formatted)
    
    # Calculate tokens (divide by 4 as an upper bound heuristic)
    total_prompt_tokens = len(system_prompt) // 4
    
    assert total_prompt_tokens <= MAX_SYNTHESIS_INPUT_TOKENS, (
        f"Prompt blew up to {total_prompt_tokens} tokens, exceeding budget of {MAX_SYNTHESIS_INPUT_TOKENS}"
    )
