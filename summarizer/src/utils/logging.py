import warnings
# import logging
from langchain_core._api import LangChainDeprecationWarning


def set_logger(ignore_warnings=True):

    if not ignore_warnings:
        return

    print("setting logging...\n")

    # logger = logging.getLogger("langchain_core")
    # logger.setLevel(logging.ERROR)

    # warnings.filterwarnings("ignore", category=logging.warning)
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
