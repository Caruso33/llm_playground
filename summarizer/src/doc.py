from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def get_docs(url) -> List[Document]:
    loader = WebBaseLoader(url)

    docs = loader.load()

    return docs


def get_split_docs(docs: List[Document]):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    return split_docs
