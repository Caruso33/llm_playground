from typing import List

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document


def transcribe(urls: List[str]) -> List[Document]:

    url_docs = []

    for url in urls:

        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        docs = loader.load()

        url_docs.append(docs)

    return url_docs
