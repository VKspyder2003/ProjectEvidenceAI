import pytest
import httpx
from src.mcp_server.github_server import (
    get_recent_pull_requests,
    get_pr_diff,
    search_issues,
    read_repository_file
)
import src.mcp_server.github_server as gh_server
from src.mcp_server.models import ToolResult

def handler(request: httpx.Request):
    path = request.url.path
    if path == "/repos/test/repo/pulls":
        return httpx.Response(200, json=[
            {
                "number": 1,
                "title": "Test PR",
                "user": {"login": "testuser"},
                "state": "open",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
                "html_url": "http://github.com/test/repo/pull/1",
                "merged_at": None
            },
            {
                "number": 2,
                "title": "Closed but Unmerged PR",
                "user": {"login": "testuser"},
                "state": "closed",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
                "html_url": "http://github.com/test/repo/pull/2",
                "merged_at": None
            }
        ])
    elif path == "/repos/test/notfound/pulls":
        return httpx.Response(404, json={"message": "Not Found"})
    elif path == "/repos/test/ratelimit/pulls":
        return httpx.Response(403, json={"message": "API rate limit exceeded"}, headers={
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1600000000"
        })
    elif path == "/repos/test/repo/pulls/1/files":
        return httpx.Response(200, json=[
            {
                "filename": "file1.txt",
                "status": "modified",
                "additions": 10,
                "deletions": 5,
                "changes": 15,
                "patch": "@@ -1 +1 @@\\n-old\\n+new"
            },
            {
                "filename": "huge_file.txt",
                "status": "modified",
                "additions": 1000,
                "deletions": 0,
                "changes": 1000,
                "patch": "A" * 10000
            },
            {
                "filename": "missing_patch_file.bin",
                "status": "added",
                "additions": 0,
                "deletions": 0,
                "changes": 0
            }
        ])
    elif path == "/search/issues":
        if "empty" in str(request.url.query):
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={
            "items": [
                {
                    "number": 2,
                    "title": "Test Issue",
                    "state": "open",
                    "html_url": "http://github.com/test/repo/issues/2",
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-02T00:00:00Z",
                    "comments": 0
                }
            ]
        })
    elif path == "/repos/test/repo/contents/README.md":
        import base64
        return httpx.Response(200, json={
            "path": "README.md",
            "size": 12,
            "content": base64.b64encode(b"Hello World!").decode('ascii'),
            "encoding": "base64"
        })
    elif path == "/repos/test/repo/contents/binary.bin":
        import base64
        return httpx.Response(200, json={
            "path": "binary.bin",
            "size": 3,
            "content": base64.b64encode(bytes([255, 254, 255])).decode('ascii'),
            "encoding": "base64"
        })
    return httpx.Response(404, json={"message": "Not Found"})

@pytest.fixture(autouse=True)
def setup_mock_client():
    transport = httpx.MockTransport(handler)
    gh_server._client = httpx.AsyncClient(transport=transport, base_url="https://api.github.com")
    yield
    gh_server._client = None

@pytest.mark.asyncio
async def test_get_recent_pull_requests_success():
    res = await get_recent_pull_requests("test", "repo")
    assert res.success is True
    assert len(res.data) == 2
    assert res.data[0]["number"] == 1
    assert res.data[0]["merged"] is False
    assert res.data[1]["number"] == 2
    assert res.data[1]["state"] == "closed"
    assert res.data[1]["merged"] is False

@pytest.mark.asyncio
async def test_get_recent_pull_requests_404():
    res = await get_recent_pull_requests("test", "notfound")
    assert res.success is False
    assert res.error.status_code == 404

@pytest.mark.asyncio
async def test_get_recent_pull_requests_rate_limit():
    res = await get_recent_pull_requests("test", "ratelimit")
    assert res.success is False
    assert res.error.status_code == 403
    assert res.error.rate_limit_remaining == 0
    assert res.error.rate_limit_reset == 1600000000

@pytest.mark.asyncio
async def test_get_pr_diff_truncation_and_missing_patch():
    res = await get_pr_diff("test", "repo", 1, max_chars_per_file=100)
    assert res.success is True
    files = res.data["files"]
    assert len(files) == 3
    assert files[0]["is_truncated"] is False
    assert files[1]["is_truncated"] is True
    assert len(files[1]["patch"]) > 100
    assert files[2]["is_binary"] is True
    assert files[2]["patch"] is None

@pytest.mark.asyncio
async def test_search_issues_empty():
    res = await search_issues("test", "repo", "empty")
    assert res.success is True
    assert len(res.data) == 0

@pytest.mark.asyncio
async def test_read_repository_file():
    res = await read_repository_file("test", "repo", "README.md")
    assert res.success is True
    assert res.data["content"] == "Hello World!"

@pytest.mark.asyncio
async def test_read_repository_file_binary():
    res = await read_repository_file("test", "repo", "binary.bin")
    assert res.success is False
    assert res.error.status_code == 400
    assert "binary" in res.error.message
