import re
from pydantic import BaseModel
from typing import Optional

class GuardrailResult(BaseModel):
    is_allowed: bool
    reason: Optional[str] = None

# 1. Prompt injection attempts
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|prompts)\b",
    r"(?i)\bforget\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|prompts)\b",
    r"(?i)\bdisregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|prompts)\b",
]

# 2. System-prompt extraction attempts
SYSTEM_PROMPT_EXTRACTION_PATTERNS = [
    r"(?i)\b(?:what|show|print|reveal|tell|read|display).*?\b(?:system prompt|instructions|core instructions|rules)\b",
    r"(?i)\bhow\s+are\s+you\s+(?:configured|programmed|instructed)\b",
]

# 3. Secret/API-key/environment-variable extraction attempts
SECRET_EXTRACTION_PATTERNS = [
    r"(?i)\b(?:api_key|apikey|api\s+key(?:s)?|secret(?:s)?|token(?:s)?|password(?:s)?|credential(?:s)?)\b",
    r"(?i)\b(?:aws_access_key|aws_secret_key|github_token|gh_token|openai_api_key)\b",
    r"(?i)\b(?:env|environment)\s+(?:variable(?:s)?|var(?:s)?)\b",
]

# 4. Attempts to bypass the GitHub-only tool boundary
BOUNDARY_BYPASS_PATTERNS = [
    r"(?i)\b(?:search\s+google|search\s+the\s+web|browse\s+internet)\b",
    r"(?i)\b(?:run\s+command|execute\s+script|run\s+bash|exec\b)",
    r"(?i)http(?:s)?://(?:(?!github\.com).)+", # Match non-github URLs
]

# 5. Attempts to override evidence-grounding requirements
EVIDENCE_OVERRIDE_PATTERNS = [
    r"(?i)\b(?:make\s+up|invent|fabricate|hallucinate)\b.*?\b(?:pr|pull request|issue|commit|file|evidence)\b",
    r"(?i)\b(?:don'?t\s+use|ignore|without)\b.*?\bevidence\b",
]


def validate_input(query: str) -> GuardrailResult:
    """
    Validates the user query deterministically without using an LLM.
    Returns a GuardrailResult indicating if the query is allowed or blocked.
    """
    
    # 1. Prompt injection attempts
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query):
            return GuardrailResult(is_allowed=False, reason="Prompt injection attempt detected.")

    # 2. System-prompt extraction attempts
    for pattern in SYSTEM_PROMPT_EXTRACTION_PATTERNS:
        if re.search(pattern, query):
            return GuardrailResult(is_allowed=False, reason="System prompt extraction attempt detected.")

    # 3. Secret extraction
    for pattern in SECRET_EXTRACTION_PATTERNS:
        if re.search(pattern, query):
            return GuardrailResult(is_allowed=False, reason="Secret/credential extraction attempt detected.")

    # 4. Boundary bypass
    for pattern in BOUNDARY_BYPASS_PATTERNS:
        if re.search(pattern, query):
            return GuardrailResult(is_allowed=False, reason="Tool boundary bypass attempt detected.")

    # 5. Evidence grounding override
    for pattern in EVIDENCE_OVERRIDE_PATTERNS:
        if re.search(pattern, query):
            return GuardrailResult(is_allowed=False, reason="Evidence override attempt detected.")

    return GuardrailResult(is_allowed=True)
