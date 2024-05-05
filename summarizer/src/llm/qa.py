from typing import List

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .model import get_llm


def get_qa_chain(split_docs: List[Document]) -> RetrievalQA:

    # llm = _get_llm()

    # prompt_template = """Answer the following question:
    # {question}
    # ANSWER:"""
    # prompt = PromptTemplate.from_template(prompt_template)

    # llm_chain = llm | prompt

    # return llm_chain

    # Combine doc
    # Build an index
    embeddings = HuggingFaceEmbeddings()
    # embeddings = OpenAIEmbeddings()

    vectordb = FAISS.from_texts(split_docs, embeddings)

    # Build a QA chain
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )

    return chain
