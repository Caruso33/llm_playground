from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from .weburl import load_weburl


def get_docs(url: str) -> List[Document]:

    if url.startswith("http"):
        loader = load_weburl(url)

    elif url.endswith(".md"):
        loader = TextLoader(url)

    else:
        raise ValueError(f"Unsupported URL format: {url}")

    docs = loader.load()

    return docs


def get_split_docs(docs: List[Document]):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    return split_docs
