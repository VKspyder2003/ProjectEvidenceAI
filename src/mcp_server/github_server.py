import httpx
import base64
import contextlib
from typing import Optional
from fastmcp import FastMCP
try:
    from .config import config
    from .models import (
        ToolResult, GitHubError, PullRequestModel, 
        FileDiffMetadata, PRDiffResult, IssueModel, FileContentResult
    )
except ImportError:
    from config import config
    from models import (
        ToolResult, GitHubError, PullRequestModel, 
        FileDiffMetadata, PRDiffResult, IssueModel, FileContentResult
    )

# Global HTTP client
_client: Optional[httpx.AsyncClient] = None

@contextlib.asynccontextmanager
async def mcp_lifespan(server):
    """Lifecycle manager for the FastMCP server to handle the HTTPX client."""
    global _client
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": config.github_user_agent,
        "X-GitHub-Api-Version": config.github_api_version
    }
    if config.github_token:
        headers["Authorization"] = f"Bearer {config.github_token}"
        
    _client = httpx.AsyncClient(
        base_url=config.github_api_base_url,
        headers=headers,
        timeout=config.github_api_timeout
    )
    yield
    if _client:
        await _client.aclose()
        _client = None

# If FastMCP supports lifespan context manager in run/init, we pass it. 
# Depending on fastmcp version, it might accept it in FastMCP() or mcp.run()
# We will use it if the library supports it, or rely on lazy init.
mcp = FastMCP("GitHub") 

# Fallback lazy init if lifespan is not triggered during tests/usage
async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": config.github_user_agent,
            "X-GitHub-Api-Version": config.github_api_version
        }
        if config.github_token:
            headers["Authorization"] = f"Bearer {config.github_token}"
            
        _client = httpx.AsyncClient(
            base_url=config.github_api_base_url,
            headers=headers,
            timeout=config.github_api_timeout
        )
    return _client

def handle_error(e: httpx.HTTPError) -> ToolResult:
    if isinstance(e, httpx.HTTPStatusError):
        response = e.response
        rate_limit_remaining = int(response.headers.get("x-ratelimit-remaining", -1))
        rate_limit_reset = int(response.headers.get("x-ratelimit-reset", -1))
        
        rate_limit_remaining = rate_limit_remaining if rate_limit_remaining != -1 else None
        rate_limit_reset = rate_limit_reset if rate_limit_reset != -1 else None
        
        try:
            message = response.json().get("message", str(e))
        except Exception:
            message = str(e)
            
        return ToolResult(
            success=False,
            error=GitHubError(
                status_code=response.status_code,
                message=message,
                rate_limit_remaining=rate_limit_remaining,
                rate_limit_reset=rate_limit_reset
            )
        )
    return ToolResult(
        success=False,
        error=GitHubError(
            status_code=500,
            message=str(e)
        )
    )

@mcp.tool()
async def get_recent_pull_requests(
    repo_owner: str, 
    repo_name: str, 
    count: int = 5, 
    state: str = "all", 
    sort: str = "updated", 
    direction: str = "desc"
) -> ToolResult:
    """
    Returns recent pull requests for a given repository.
    """
    client = await get_client()
    count = min(count, 100)
    
    try:
        response = await client.get(
            f"/repos/{repo_owner}/{repo_name}/pulls",
            params={
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": count
            }
        )
        response.raise_for_status()
        
        prs = []
        for pr in response.json():
            prs.append(PullRequestModel(
                number=pr["number"],
                title=pr["title"],
                author=pr["user"]["login"],
                state=pr["state"],
                merged=pr.get("merged_at") is not None,
                created_at=pr["created_at"],
                updated_at=pr["updated_at"],
                merged_at=pr.get("merged_at"),
                html_url=pr["html_url"]
            ))
            
        return ToolResult(success=True, data=[pr.model_dump() for pr in prs])
    except httpx.HTTPError as e:
        return handle_error(e)


@mcp.tool()
async def get_pr_diff(
    repo_owner: str, 
    repo_name: str, 
    pr_number: int, 
    max_files: int = 20, 
    max_chars_per_file: int = 8000
) -> ToolResult:
    """
    Returns the diff for a specific pull request, including added/removed lines.
    """
    client = await get_client()
    
    try:
        response = await client.get(
            f"/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/files",
            params={"per_page": 100}
        )
        response.raise_for_status()
        
        all_files = response.json()
        analyzed_files = []
        
        for file_data in all_files[:max_files]:
            patch = file_data.get("patch")
            is_truncated = False
            is_binary = False
            
            if patch is None:
                is_binary = True
            elif len(patch) > max_chars_per_file:
                patch = patch[:max_chars_per_file] + "\n...[TRUNCATED]"
                is_truncated = True
                
            analyzed_files.append(FileDiffMetadata(
                filename=file_data["filename"],
                status=file_data["status"],
                additions=file_data.get("additions", 0),
                deletions=file_data.get("deletions", 0),
                changes=file_data.get("changes", 0),
                patch=patch,
                is_truncated=is_truncated,
                is_binary=is_binary
            ))
            
        result = PRDiffResult(
            pr_number=pr_number,
            files=analyzed_files,
            total_files_analyzed=len(analyzed_files),
            files_omitted=max(0, len(all_files) - max_files)
        )
        
        return ToolResult(success=True, data=result.model_dump())
    except httpx.HTTPError as e:
        return handle_error(e)


@mcp.tool()
async def search_issues(
    repo_owner: str, 
    repo_name: str, 
    query: str, 
    state: str = "open", 
    include_comments: bool = False, 
    max_results: int = 20
) -> ToolResult:
    """
    Searches issues in a repository (explicitly excludes pull requests).
    """
    client = await get_client()
    
    q = f"repo:{repo_owner}/{repo_name} is:issue {query}"
    if state != "all":
        q += f" state:{state}"
        
    try:
        response = await client.get(
            "/search/issues",
            params={"q": q, "per_page": max_results}
        )
        response.raise_for_status()
        
        items = response.json().get("items", [])
        issues = []
        
        for item in items:
            issue = IssueModel(
                number=item["number"],
                title=item["title"],
                state=item["state"],
                html_url=item["html_url"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                body=item.get("body")
            )
            
            if include_comments and item.get("comments", 0) > 0:
                comments_resp = await client.get(
                    f"/repos/{repo_owner}/{repo_name}/issues/{item['number']}/comments",
                    params={"per_page": 5}
                )
                if comments_resp.status_code == 200:
                    comments = []
                    for c in comments_resp.json():
                        comments.append({
                            "user": c["user"]["login"],
                            "created_at": c["created_at"],
                            "body": c["body"]
                        })
                    issue.comments = comments
                    
            issues.append(issue)
            
        return ToolResult(success=True, data=[i.model_dump() for i in issues])
    except httpx.HTTPError as e:
        return handle_error(e)


@mcp.tool()
async def read_repository_file(
    repo_owner: str, 
    repo_name: str, 
    file_path: str, 
    branch: str = "main", 
    max_chars: int = 50000
) -> ToolResult:
    """
    Reads the contents of a file in the repository.
    """
    client = await get_client()
    
    try:
        response = await client.get(
            f"/repos/{repo_owner}/{repo_name}/contents/{file_path}",
            params={"ref": branch}
        )
        response.raise_for_status()
        
        data = response.json()
        
        if isinstance(data, list):
            return ToolResult(
                success=False, 
                error=GitHubError(status_code=400, message="Path is a directory, not a file")
            )
            
        if data.get("encoding") == "base64":
            content_bytes = base64.b64decode(data["content"])
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return ToolResult(
                    success=False,
                    error=GitHubError(status_code=400, message="File is binary and cannot be decoded as utf-8")
                )
        else:
            content = data.get("content", "")
            
        is_truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            is_truncated = True
            
        result = FileContentResult(
            file_path=data["path"],
            content=content,
            size=data["size"],
            is_truncated=is_truncated,
            encoding="utf-8"
        )
        
        return ToolResult(success=True, data=result.model_dump())
    except httpx.HTTPError as e:
        return handle_error(e)

if __name__ == "__main__":
    mcp.run()
