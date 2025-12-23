#!/usr/bin/env python3
"""
Entry point for running the Regression Checker Streamlit UI.

Usage:
    streamlit run run_regression_checker.py
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.regression_checker_ui import main

if __name__ == "__main__":
    main()
