#!/usr/bin/env python3
"""Retry eval with kimi model support.

Usage:
    uv run python retry_with_kimi.py <eval-file>
    
Example:
    uv run python retry_with_kimi.py logs/2026-03-30T14-14-46+00-00_aime2025_BdzUHHgw38DGvhSJVTuXWT.eval
"""
import sys
import kimi_model  # noqa: F401 - registers kimi API
from inspect_ai._cli.main import main

if __name__ == "__main__":
    # 把 eval-retry 参数传给 inspect CLI
    sys.argv = ["inspect", "eval-retry"] + sys.argv[1:]
    main()
