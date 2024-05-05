from langchain_community.document_loaders import WebBaseLoader
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PlaywrightURLLoader


def load_weburl(url: Union[str, List[str]]) -> List[Document]:

    # https://python.langchain.com/docs/integrations/document_loaders/web_base/
    loader = WebBaseLoader(url)

    # https://python.langchain.com/docs/integrations/document_loaders/url/
    loader = PlaywrightURLLoader(urls=url, remove_selectors=["header", "footer"])

    docs = loader.load()

    return docs
