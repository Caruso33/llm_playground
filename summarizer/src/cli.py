import argparse


def parse_cli():

    parser = argparse.ArgumentParser(description="Summarizr - Summarize any content")

    parser.add_argument("--url", "-u", required=True, help="URL to summarize content")
    parser.add_argument(
        "--length", "-l", required=False, help="Length of summary in chars"
    )

    args = parser.parse_args()

    return args
