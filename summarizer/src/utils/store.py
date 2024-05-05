import json
import os
import pickle
from typing import List

from langchain_core.documents import Document


def save_json(
    docs: List[Document],
    json_data_path=os.path.join(
        os.path.join(os.getcwd(), "downloads"), "documents.json"
    ),
):

    # Assuming `docs` is a list of Document objects
    # that can be serialized to JSON.

    # To store documents as JSON
    with open(json_data_path, "w") as f:
        json.dump([doc.to_dict() for doc in docs], f)


def load_json(
    json_data_path=os.path.join(
        os.path.join(os.getcwd(), "downloads"), "documents.json"
    )
) -> List[Document]:
    # To load documents from JSON
    with open(json_data_path, "r") as f:
        documents_data = json.load(f)

        docs = [Document.from_dict(doc) for doc in documents_data]

        return docs


def save_pickle(
    docs: List[Document],
    pickle_data_path=os.path.join(
        os.path.join(os.getcwd(), "downloads"), "documents.pkl"
    ),
):

    # To store documents using pickle
    with open(pickle_data_path, "wb") as f:
        pickle.dump(docs, f)


def load_pickle(
    pickle_data_path=os.path.join(
        os.path.join(os.getcwd(), "downloads"), "documents.pkl"
    )
) -> List[Document]:
    # To load documents using pickle
    with open(pickle_data_path, "rb") as f:
        docs = pickle.load(f)

        return docs
