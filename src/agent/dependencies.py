import os
from dataclasses import dataclass
from typing import Any
from langchain_core.language_models import BaseChatModel

@dataclass
class AgentDependencies:
    llm: BaseChatModel
    mcp_server: Any

def get_llm() -> BaseChatModel:
    """Instantiate the configured LLM based on environment variables."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    env_model = os.getenv("LLM_MODEL")
    if not env_model or env_model == "llama3":
        if provider == "groq":
            model_name = "qwen/qwen3.8-27b"
        elif provider in ("gemini", "google"):
            model_name = "gemini-1.5-pro"
        else:
            model_name = "llama3"
    else:
        model_name = env_model

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model=model_name, max_tokens=2000)
        except ImportError:
            raise ImportError("Please install langchain-groq for Groq support")
    elif provider == "gemini" or provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name)
        except ImportError:
            raise ImportError("Please install langchain-google-genai for Gemini support")
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name)
        except ImportError:
            try:
                # Fallback
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(model=model_name)
            except ImportError:
                raise ImportError("Please install langchain-ollama for Ollama support")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_agent_dependencies() -> AgentDependencies:
    """Initialize and return the infrastructure dependencies."""
    from src.mcp_server.github_server import mcp
    
    llm = get_llm()
    return AgentDependencies(
        llm=llm,
        mcp_server=mcp
    )
