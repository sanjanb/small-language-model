#!/usr/bin/env python3
"""Main entry point for the document extraction system."""
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from document_extractor.cli import cli

if __name__ == '__main__':
    cli()