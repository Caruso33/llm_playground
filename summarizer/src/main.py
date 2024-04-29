from src.cli import parse_cli
from src.doc import get_docs, get_split_docs
from src.llm import ChainType, get_chain

# templates based on
# https://python.langchain.com/docs/use_cases/summarization/
# https://python.langchain.com/docs/use_cases/web_scraping/


def main():
    print(
        """
                                          .__                
  ________ __  _____   _____ _____ _______|__|______________ 
 /  ___/  |  \/     \ /     \\__  \\_  __ \  \___   /\_  __ \\
 \___ \|  |  /  Y Y  \  Y Y  \/ __ \|  | \/  |/    /  |  | \/
/____  >____/|__|_|  /__|_|  (____  /__|  |__/_____ \ |__|   
     \/            \/      \/     \/               \/        
"""
    )

    args = parse_cli()

    # Access the value of the URL argument
    url = args.url
    length = args.length

    if url is None:
        raise ValueError("No URL provided. Please provide a URL.")

    print(f"url {url}\n")
    if length is not None:
        print(f"length {length} words\n")

    chain_type = ChainType.MAP

    docs = get_docs(url)

    if chain_type == ChainType.MAP:
        docs = get_split_docs(docs)

    # print(f"docs {type(docs[0])} {docs[:2]}\n")
    # print(f"len(docs) {len(docs)}\n")

    chain = get_chain(chain_type, length)

    results = chain.invoke(docs)

    print(f"results {results['output_text']}\n")