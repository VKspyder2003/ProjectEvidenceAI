from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class GitHubConfig(BaseSettings):
    github_token: Optional[str] = None
    github_api_base_url: str = "https://api.github.com"
    github_user_agent: str = "AgenticRAG-GitHub-MCP/1.0"
    github_api_timeout: int = 30
    github_api_version: str = "2022-11-28"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

config = GitHubConfig()
