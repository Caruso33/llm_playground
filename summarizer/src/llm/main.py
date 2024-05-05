import os

from dotenv import load_dotenv
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain

from .summarize import get_map_chain, get_stuff_chain, get_summarize_chain
from .types import ChainType, SummarizeType

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_CONTEXT_LENGTH = os.getenv("MODEL_CONTEXT_LENGTH")


def get_chain(
    chain_type: ChainType,
    length,
    objective,
    summarize_type=SummarizeType.STUFF,
) -> BaseCombineDocumentsChain:

    if chain_type == ChainType.SUMMARIZE:
        return get_summarize_chain(summarize_type)

    elif chain_type == ChainType.STUFF:
        return get_stuff_chain()

    elif chain_type == ChainType.MAP:
        return get_map_chain(length, objective)

    else:
        raise ValueError(
            f"chain_type must be 'summarize', 'stuff', or 'map_reduce', but got {chain_type}"
        )
