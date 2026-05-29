"""Pytest bootstrap: make the src/ layout importable without an install step."""
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
