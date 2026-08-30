import json
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState, Evidence
from .dependencies import AgentDependencies
from .prompts import SYNTHESIZER_SYSTEM_PROMPT

def format_evidence_for_prompt(evidence_list: List[Evidence]) -> str:
    """
    Formats the retrieved evidence into a clear boundary for the LLM.
    """
    if not evidence_list:
        return "No evidence retrieved."
        
    formatted_blocks = []
    for i, ev in enumerate(evidence_list, 1):
        block = [f"SOURCE {i}"]
        block.append(f"Type: {ev.source_type}")
        block.append(f"ID: {ev.source_id}")
        if ev.url:
            block.append(f"URL: {ev.url}")
            
        block.append("\nEvidence:")
        # Convert content to a readable JSON string or dict
        block.append(json.dumps(ev.content))
        
        formatted_blocks.append("\n".join(block))
        
    return "\n\n---\n\n".join(formatted_blocks)

async def synthesizer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Synthesizer node that generates the final grounded answer based on collected evidence.
    """
    evidence = state.get("retrieved_evidence", [])
    query = state.get("query", "")
    
    # Short-circuit if no evidence exists
    if not evidence:
        return {
            "draft_response": "I couldn't find sufficient repository evidence to answer this question.",
            "error": None
        }

    configurable = config.get("configurable", {})
    dependencies: AgentDependencies = configurable.get("dependencies")
    if not dependencies:
        raise ValueError("AgentDependencies not found in RunnableConfig['configurable']")

    llm = dependencies.llm
    
    evidence_context = format_evidence_for_prompt(evidence)
    
    system_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(
        evidence_context=evidence_context
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    try:
        # Standard text response for final answer
        response = await llm.ainvoke(messages)
        return {
            "draft_response": response.content,
            "error": None
        }
    except Exception as e:
        return {
            "draft_response": None,
            "error": f"Synthesis failed: {str(e)}"
        }
