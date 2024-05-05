from typing import List

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_qa_chain(docs: List[Document]):
    # Combine doc
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)

    # Split them
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    # Build an index
    embeddings = HuggingFaceEmbeddings()
    # embeddings = OpenAIEmbeddings()

    vectordb = FAISS.from_texts(splits, embeddings)

    # Build a QA chain
    chain = RetrievalQA.from_chain_type(
        llm=get_qa_chain(),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )

    return chain
