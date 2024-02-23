import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

from pathlib import Path

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent
