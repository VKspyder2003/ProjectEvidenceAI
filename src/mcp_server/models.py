from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class GitHubError(BaseModel):
    status_code: int = Field(..., description="HTTP status code from GitHub API")
    message: str = Field(..., description="Error message")
    rate_limit_remaining: Optional[int] = Field(None, description="Remaining rate limit requests")
    rate_limit_reset: Optional[int] = Field(None, description="Unix timestamp when rate limit resets")

class ToolResult(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    data: Any = Field(None, description="The returned data if successful. Could be empty if no results.")
    error: Optional[GitHubError] = Field(None, description="Error details if unsuccessful")
    
class PullRequestModel(BaseModel):
    number: int
    title: str
    author: str
    state: str
    merged: bool
    created_at: str
    updated_at: str
    merged_at: Optional[str] = None
    html_url: str

class FileDiffMetadata(BaseModel):
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: Optional[str] = None
    is_truncated: bool = False
    is_binary: bool = False

class PRDiffResult(BaseModel):
    pr_number: int
    files: List[FileDiffMetadata]
    total_files_analyzed: int
    files_omitted: int

class IssueModel(BaseModel):
    number: int
    title: str
    state: str
    html_url: str
    created_at: str
    updated_at: str
    body: Optional[str] = None
    comments: Optional[List[Dict[str, Any]]] = None
    
class FileContentResult(BaseModel):
    file_path: str
    content: str
    size: int
    html_url: str
    is_truncated: bool = False
    encoding: str = "utf-8"
