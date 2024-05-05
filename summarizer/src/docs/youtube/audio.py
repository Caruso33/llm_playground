import os
from typing import List

from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)
from langchain_core.documents import Document

# https://python.langchain.com/docs/integrations/document_loaders/youtube_audio/


def load_and_save_audio(
    urls: List[str], save_dir=os.path.join(os.getcwd(), "downloads")
) -> List[Document]:
    # set a flag to switch between local and remote parsing
    # change this to True if you want to use local parsing
    local = True

    # Transcribe the videos to text
    if local:
        loader = GenericLoader(
            YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal()
        )
    else:
        loader = GenericLoader(
            YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser()
        )

    docs = loader.load()

    return docs
