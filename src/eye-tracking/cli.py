"""This handles the CLI for the Eye Tracking application."""

import sys
import argparse
import logging


def main() -> None:
    """
    A simple CLI tool that converts docx document to Alstom SwRS format (Sphinx).

    Args:
        input (str)         : The path of the docx document.
        output (str)        : The destination path

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="A simple CLI tool")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be vebose",
        action="count",
        dest="loglevel",
        default=0,
    )

    args = parser.parse_args()
    # Logging level start at Warning (level = 30)
    # Each -v sub 10 to logging level (minimum is 0)
    loglevel = max(logging.WARNING - (args.loglevel * 10), 0)
    # Force UTF-8 encoding cause in certain Windows OS, 
    # it may be other encoder by default (case encounter : 'cp1252')
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
    logging.basicConfig(
        level=loglevel, format="%(message)s"
    )
