from langchain_community.llms.ollama import Ollama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from .env import LLM_PROVIDER, MODEL_CONTEXT_LENGTH, MODEL_NAME


def get_llm() -> ChatOpenAI | ChatGroq | Ollama:

    if MODEL_NAME is None:
        raise ValueError("MODEL_NAME must be set")
    if MODEL_CONTEXT_LENGTH is None:
        raise ValueError("MODEL_CONTEXT_LENGTH must be set")

    if LLM_PROVIDER == "openai":
        llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
    elif LLM_PROVIDER == "groq":
        llm = ChatGroq(temperature=0, model_name=MODEL_NAME)
    elif LLM_PROVIDER == "ollama":
        llm = Ollama(temperature=0, model=MODEL_NAME)
    else:
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'groq' or `ollama`, but got {LLM_PROVIDER}"
        )

    print(
        f"Using {LLM_PROVIDER}'s {MODEL_NAME} and a context length of {MODEL_CONTEXT_LENGTH}\n"
    )

    return llm
