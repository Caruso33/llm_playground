from src.docs import get_docs, get_split_docs, get_split_docs_recursively
from src.llm import ChainType, get_chain, get_qa_chain
from src.utils import parse_cli, print_splashscreen

# templates based on
# https://python.langchain.com/docs/use_cases/summarization/
# https://python.langchain.com/docs/use_cases/web_scraping/


def main():
    print_splashscreen()

    args = parse_cli()
    [url, length, objective] = [args["url"], args["length"], args["objective"]]

    chain_type = ChainType.MAP

    docs = get_docs(url)

    # print(f"docs:\t\t\t{type(docs)} {type(docs[0])}")
    # print(f"len docs:\t\t\t{len(docs)}\n")

    if chain_type == ChainType.MAP:
        split_docs = get_split_docs(docs)

    # print(f"docs splitted:\t\t{type(split_docs)} {type(split_docs[0])}")
    # print(f"len docs splitted:\t{len(split_docs)}\n")

    # Summarize

    chain = get_chain(chain_type, length, objective)
    results = chain.invoke(split_docs)
    summary = results["output_text"]
    print(f"Summary:\t\t{summary}\n")

    # QA

    split_text = get_split_docs_recursively(docs)
    qa_chain = get_qa_chain(split_text)
    results = qa_chain.invoke("who was doing all that?")
    answer = results
    print(f"Answer:\t\t{answer}\n")
