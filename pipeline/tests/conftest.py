"""
Pytest configuration for M01–M10 tests.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path for old AETHERIS imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
