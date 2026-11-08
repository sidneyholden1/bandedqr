"""Make ``src/`` importable so the suite runs from a checkout without installing.

If bandedqr *is* installed (``pip install -e .``) that copy wins; this only fills
in the `make && pytest` path.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
