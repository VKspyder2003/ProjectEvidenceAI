import asyncio
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastmcp import Client

from src.mcp_server.github_server import mcp


async def main():
    async with Client(mcp) as client:
        # 1. Discover MCP tools
        tools = await client.list_tools()

        print("\nAvailable MCP tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # 2. Call get_recent_pull_requests
        print("\n--- Testing get_recent_pull_requests ---")
        pr_result = await client.call_tool(
            "get_recent_pull_requests",
            {
                "repo_owner": "octocat",
                "repo_name": "Hello-World",
                "count": 5,
                "state": "all",
            },
        )
        print(json.dumps(pr_result.structured_content, indent=2))

        # 3. Call get_pr_diff
        print("\n--- Testing get_pr_diff ---")
        diff_result = await client.call_tool(
            "get_pr_diff",
            {
                "repo_owner": "octocat",
                "repo_name": "Hello-World",
                "pr_number": 1,
                "max_files": 5,
                "max_chars_per_file": 2000,
            },
        )
        print(json.dumps(diff_result.structured_content, indent=2))

        # 4. Call search_issues
        print("\n--- Testing search_issues ---")
        issues_result = await client.call_tool(
            "search_issues",
            {
                "repo_owner": "octocat",
                "repo_name": "Hello-World",
                "query": "hello",
                "state": "all",
                "include_comments": False,
                "max_results": 2,
            },
        )
        print(json.dumps(issues_result.structured_content, indent=2))

        # 5. Call read_repository_file
        print("\n--- Testing read_repository_file ---")
        file_result = await client.call_tool(
            "read_repository_file",
            {
                "repo_owner": "octocat",
                "repo_name": "Hello-World",
                "file_path": "README", # Octocat's Hello-World has a file literally named README
                "branch": "master",
            },
        )
        print(json.dumps(file_result.structured_content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
