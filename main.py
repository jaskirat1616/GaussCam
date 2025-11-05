"""
GaussCam Main Entry Point

Launch the GaussCam GUI application.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ui.main_window import main

if __name__ == "__main__":
    main()

