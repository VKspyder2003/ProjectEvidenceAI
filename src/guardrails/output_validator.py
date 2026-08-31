import re
from typing import List, Optional
from pydantic import BaseModel
from src.agent.state import Evidence

class ValidatedCitation(BaseModel):
    source_id: str
    url: str
    is_valid: bool
    reason: Optional[str] = None

class OutputValidationResult(BaseModel):
    valid: bool
    violations: List[str]
    validated_citations: List[ValidatedCitation]

# Matches [link text](url)
CITATION_REGEX = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

def _extract_repo_from_url(url: str) -> Optional[str]:
    match = re.match(r'https?://github\.com/([^/]+/[^/]+)', url)
    if match:
        return match.group(1).lower()
    return None

def validate_output(response: str, evidence: List[Evidence]) -> OutputValidationResult:
    violations = []
    validated_citations = []
    
    citations = CITATION_REGEX.findall(response)
    
    # 8. Missing citations
    if not citations and evidence:
        fallback_phrases = [
            "could not determine the answer",
            "couldn't find sufficient repository evidence",
            "i cannot answer",
            "no evidence retrieved"
        ]
        if not any(phrase in response.lower() for phrase in fallback_phrases):
            violations.append("Response contains no citations despite available citable evidence.")

    evidence_by_url = {e.url: e for e in evidence if e.url}
    evidence_by_id = {e.source_id: e for e in evidence}
    
    evidence_repos = set()
    for e in evidence:
        if e.url:
            repo = _extract_repo_from_url(e.url)
            if repo:
                evidence_repos.add(repo)

    for source_id, url in citations:
        is_valid = True
        reason = None
        
        if url in evidence_by_url:
            # Perfect URL match
            pass
        else:
            # Check by source ID
            ev = evidence_by_id.get(source_id)
            if ev:
                # 7. Mismatched URL
                is_valid = False
                reason = f"Citation URL '{url}' does not match the evidence URL '{ev.url}'."
                violations.append(reason)
            else:
                # Not in evidence at all
                is_valid = False
                
                cited_repo = _extract_repo_from_url(url)
                if cited_repo and evidence_repos and cited_repo not in evidence_repos:
                    # 6. Wrong repository
                    reason = f"Citation references a different repository '{cited_repo}'."
                else:
                    # 4 & 5. Fabricated source / issue
                    reason = f"Source '{source_id}' was cited but was not present in retrieved evidence."
                
                violations.append(reason)

        validated_citations.append(ValidatedCitation(
            source_id=source_id,
            url=url,
            is_valid=is_valid,
            reason=reason
        ))

    return OutputValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        validated_citations=validated_citations
    )
