import argparse


def parse_cli():

    parser = argparse.ArgumentParser(description="Summarizr - Summarize any content")

    parser.add_argument("--url", "-u", required=True, help="URL to summarize content")
    parser.add_argument(
        "--length", "-l", required=False, help="Length of summary in chars"
    )
    parser.add_argument(
        "--objective", "-o", required=False, help="Objective for summary prompt"
    )

    args = parser.parse_args()

    # Access the value of the URL argument
    url = args.url
    length = args.length
    objective = args.objective

    if url is None:
        raise ValueError("No URL provided. Please provide an URL.")

    print(f"url\t\t\t{url}\n")
    if length is not None:
        print(f"length\t\t\t{length} words\n")
    if objective is not None:
        print(f"objective\t\t{objective}\n")

    return {
        "url": url,
        "length": length,
        "objective": objective,
    }
