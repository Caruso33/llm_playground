from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

from .env import MODEL_CONTEXT_LENGTH
from .model import get_llm
from .types import SummarizeType


def get_summarize_chain(chain_type: SummarizeType) -> BaseCombineDocumentsChain:

    llm = get_llm()

    chain = load_summarize_chain(llm, chain_type=chain_type)

    return chain


def get_stuff_chain():

    llm = get_llm()

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


def get_map_chain(length, objective, return_intermediate_steps=False):

    llm = get_llm()

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
        reduce_template += f"\nKeep the answer within {length} words"

    if objective is not None:
        reduce_template += f"\nPlease pay attention to also this objective, it takes preceding over everything: {objective}"
    # reduce_template += "\nHelpful Answer:"

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
