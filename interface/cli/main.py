import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Create logs directory if it doesn't exist
logs_dir = project_root / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'logs' / 'cli.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from interface.cli.commands import DRNCLI
    from engine.license import LicenseManager
    from engine.config import Config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def check_environment() -> bool:
    """Verify CLI environment is properly set up"""
    # Check Python version
    if sys.version_info < (3, 9):
        logger.error("Python 3.9 or higher is required")
        return False
    
    # Check required directories
    required_dirs = [
        project_root / "ai" / "models",
        project_root / "resources" / "adapters",
        project_root / "logs"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"Required directory missing: {dir_path}")
            return False
    
    # Check license
    license_manager = LicenseManager()
    if not license_manager.is_valid():
        logger.error("Invalid or expired license. Please activate your license.")
        return False
    
    return True

def main() -> int:
    """Main CLI entry point"""
    try:
        # Verify environment
        if not check_environment():
            return 1
        
        # Load configuration
        config = Config()
        logger.info(f"DRN.today CLI v{config.get_version()}")
        
        # Initialize and run CLI
        cli = DRNCLI()
        return cli.run()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
