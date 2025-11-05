"""
GaussCam Main Entry Point

Launch the GaussCam GUI application.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging before importing other modules
from backend.utils.logging_config import setup_logging

# Setup logging first
logger = setup_logging(
    log_level=getattr(__import__('logging'), 'INFO'),
    enable_file_logging=True,
    enable_console_logging=True,
)
logger.info("Starting GaussCam application")

try:
    from backend.ui.main_window import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    logger.critical(f"Fatal error: {e}", exc_info=True)
    sys.exit(1)

