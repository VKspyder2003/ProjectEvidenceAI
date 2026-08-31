# ProjectEvidenceAI: Agentic RAG over Live GitHub Data

[![CI](https://github.com/Vishwas/ProjectEvidenceAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Vishwas/ProjectEvidenceAI/actions/workflows/ci.yml)

This repository contains Phase 4 of the Agentic RAG system connecting to live GitHub repository data using LangGraph and the Model Context Protocol (MCP).

## Setup
1. Copy `.env.example` to `.env` and fill in your `GITHUB_TOKEN`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`

## Running the Demo
You can run the interactive Streamlit UI locally to test the ProjectEvidenceAI agent:

1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your `.env` file (copy from `.env.example`)
4. Run the application:
```bash
streamlit run app.py
```
