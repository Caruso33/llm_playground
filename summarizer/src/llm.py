import enum
import os

# from langchain import hub
from dotenv import load_dotenv
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_CONTEXT_LENGTH = os.getenv("MODEL_CONTEXT_LENGTH")


class ChainType(str, enum.Enum):
    SUMMARIZE = "summarize"
    STUFF = "stuff"
    MAP = "map_reduce"


class SummarizeType(str, enum.Enum):
    STUFF = "stuff"
    MAP = "map_reduce"
    REFINE = "refine"


def get_chain(chain_type: ChainType, length=None, summarize_type=SummarizeType.STUFF):

    if chain_type == ChainType.SUMMARIZE:
        return _get_summarize_chain(summarize_type)

    elif chain_type == ChainType.STUFF:
        return _get_stuff_chain()

    elif chain_type == ChainType.MAP:
        return _get_map_chain(length)

    else:
        raise ValueError(
            f"chain_type must be 'summarize', 'stuff', or 'map_reduce', but got {chain_type}"
        )


def _get_llm() -> ChatOpenAI:

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


def _get_summarize_chain(chain_type: SummarizeType) -> BaseCombineDocumentsChain:

    llm = _get_llm()

    chain = load_summarize_chain(llm, chain_type=chain_type)

    return chain


def _get_stuff_chain():

    llm = _get_llm()

    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = llm | prompt
    # llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )

    return stuff_chain


def _get_map_chain(length=None, return_intermediate_steps=False):

    llm = _get_llm()

    # map_template = """The following is a set of documents
    # {docs}
    # Based on this list of docs, please write a concise summary.
    # CONCISE SUMMARY:"""
    map_template = """Extract the key facts out of this text. Don't include opinions. 
    {docs}
    Keep the sentences short."""

    map_prompt = PromptTemplate.from_template(map_template)

    # map_prompt = hub.pull("rlm/map-prompt")
    # map_chain = llm | map_prompt
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes.
    Solely write the summary, no introduction or conclusion. Start directly with the summary"
    """
    # Don't write anything else besides the summary.
    # Return a JSON object with a `summary` key.

    if length is not None:
        map_template += f"\nKeep the answer within {length} words"
    # map_template += "\nHelpful Answer:"

    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Note we can also get this from the prompt hub, as noted above
    # reduce_prompt = hub.pull("rlm/map-prompt")
    # reduce_chain = llm | reduce_prompt
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=int(MODEL_CONTEXT_LENGTH),
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=return_intermediate_steps,
    )

    return map_reduce_chain
