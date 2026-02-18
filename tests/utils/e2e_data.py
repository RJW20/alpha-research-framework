import argparse
import os
import sys
import unittest
from pathlib import Path

from alpha_research_framework.download import structure as data_struct

E2E_DATA_DIR = "E2E_DATA_DIR"

def set_e2e_data_dir() -> None:
    """Set environment variable E2E_DATA_DIR for end to end tests to use."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e2e-data-dir",
        help="Directory containing E2E test data"
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])

    if args.e2e_data_dir:
        os.environ[E2E_DATA_DIR] = args.e2e_data_dir

    sys.argv = [sys.argv[0]] + remaining


def require_e2e_data_dir():
    """
    Get environment variable E2E_DATA_DIR.
    
    Skips the current test if it is not set, or of it does not contain valid
    data.
    """

    data_dir = os.getenv(E2E_DATA_DIR)
    if not data_dir:
        raise unittest.SkipTest(
            "End-to-end data not provided. Use --e2e-data-dir /path/to/data."
        )

    path = Path(data_dir)
    if not path.exists():
        raise unittest.SkipTest(f"E2E data dir does not exist: {path}")
    
    metadata_path = data_struct.metadata_path(path)
    if not metadata_path.exists():
        raise unittest.SkipTest(
            f"E2E data dir does not contain required file: {metadata_path.name}"
        )

    stocks_path = data_struct.stocks_path(path)
    if not stocks_path.exists():
        raise unittest.SkipTest(
            f"E2E data dir does not contain required folder: {stocks_path.name}"
        )

    return path
