# ProjectEvidenceAI: Agentic RAG over Live GitHub Data

[![CI](https://github.com/Vishwas/ProjectEvidenceAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Vishwas/ProjectEvidenceAI/actions/workflows/ci.yml)

This repository contains Phase 4 of the Agentic RAG system connecting to live GitHub repository data using LangGraph and the Model Context Protocol (MCP).

## Setup
1. Copy `.env.example` to `.env` and fill in your `GITHUB_TOKEN`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`
