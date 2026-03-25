from typing import Any

LLM_MODEL_NAME = "Qwen3-Next-80B-A3B-Instruct"
VLLM_API_BASE = "http://192.168.0.249:44269/v1"
TEMPERATURE = 0.1
MAX_TOKENS = 1024


def build_llm_client() -> Any:
    """Build the vLLM-backed chat model used for representation generation."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        ChatOpenAI = None

    if ChatOpenAI is None:
        raise ImportError(
            "langchain-openai is required for --add_represent. "
            "Install it in the server environment before running this script."
        )

    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key="EMPTY",
        openai_api_base=VLLM_API_BASE,
        max_tokens=MAX_TOKENS,
    )
