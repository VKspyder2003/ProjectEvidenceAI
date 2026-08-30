import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class GitHubConfig(BaseModel):
    github_token: str | None = None
    github_api_base_url: str = "https://api.github.com"
    github_user_agent: str = "AgenticRAG-GitHub-MCP/1.0"
    github_api_timeout: int = 30

config = GitHubConfig(
    github_token=os.getenv("GITHUB_TOKEN"),
    github_api_base_url=os.getenv("GITHUB_API_BASE_URL", "https://api.github.com"),
    github_user_agent=os.getenv("GITHUB_USER_AGENT", "AgenticRAG-GitHub-MCP/1.0"),
    github_api_timeout=int(os.getenv("GITHUB_API_TIMEOUT", "30")),
)
