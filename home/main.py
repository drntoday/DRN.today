import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.orchestrator import SystemOrchestrator
from engine.license import LicenseManager
from engine.event_system import EventSystem
from engine.storage import SecureStorage
from interface.gui.app import DRNApplication
from interface.cli.main import CLIInterface
from modules.compliance.restrictions import GeoCompliance
from ai.models import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'drn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DRNMain:
    
    def __init__(self):
        self.orchestrator: Optional[SystemOrchestrator] = None
        self.license_manager: Optional[LicenseManager] = None
        self.event_system: Optional[EventSystem] = None
        self.storage: Optional[SecureStorage] = None
        self.model_manager: Optional[ModelManager] = None
        self.geo_compliance: Optional[GeoCompliance] = None
        
    def initialize_core_systems(self) -> bool:
        """Initialize all core system components"""
        try:
            logger.info("Initializing core systems...")
            
            # Initialize storage
            self.storage = SecureStorage()
            logger.info("✓ Secure storage initialized")
            
            # Initialize event system
            self.event_system = EventSystem()
            logger.info("✓ Event system initialized")
            
            # Initialize license manager
            self.license_manager = LicenseManager(self.storage)
            if not self.license_manager.validate_license():
                logger.error("License validation failed")
                return False
            logger.info("✓ License validated")
            
            # Initialize model manager
            self.model_manager = ModelManager()
            if not self.model_manager.load_models():
                logger.error("AI models failed to load")
                return False
            logger.info("✓ AI models loaded")
            
            # Initialize geo compliance
            self.geo_compliance = GeoCompliance()
            logger.info("✓ Geo compliance initialized")
            
            # Initialize orchestrator
            self.orchestrator = SystemOrchestrator(
                event_system=self.event_system,
                storage=self.storage,
                license_manager=self.license_manager,
                model_manager=self.model_manager,
                geo_compliance=self.geo_compliance
            )
            logger.info("✓ System orchestrator initialized")
            
            return True
            
        except Exception as e:
            logger.critical(f"System initialization failed: {str(e)}", exc_info=True)
            return False
    
    def run_gui_mode(self):
        """Launch the graphical user interface"""
        logger.info("Starting GUI mode...")
        try:
            app = DRNApplication(self.orchestrator)
            app.run()
        except Exception as e:
            logger.critical(f"GUI mode failed: {str(e)}", exc_info=True)
            sys.exit(1)
    
    def run_cli_mode(self, args: argparse.Namespace):
        """Execute command-line interface operations"""
        logger.info("Starting CLI mode...")
        try:
            cli = CLIInterface(self.orchestrator)
            cli.execute(args)
        except Exception as e:
            logger.critical(f"CLI mode failed: {str(e)}", exc_info=True)
            sys.exit(1)
    
    def run_headless_mode(self):
        """Execute headless validation for CI/CD"""
        logger.info("Starting headless validation mode...")
        try:
            # Validate core systems
            if not self.orchestrator.validate_system():
                logger.error("System validation failed")
                return False
            
            # Test license
            if not self.license_manager.validate_license():
                logger.error("License validation failed")
                return False
            
            # Test AI models
            if not self.model_manager.test_models():
                logger.error("AI model test failed")
                return False
            
            # Test geo compliance
            if not self.geo_compliance.test_compliance():
                logger.error("Geo compliance test failed")
                return False
            
            logger.info("✅ All headless validations passed")
            return True
            
        except Exception as e:
            logger.critical(f"Headless validation failed: {str(e)}", exc_info=True)
            return False
    
    def run_daemon_mode(self):
        """Start background daemon service"""
        logger.info("Starting daemon mode...")
        try:
            from engine.daemon import DRNDaemon
            daemon = DRNDaemon(self.orchestrator)
            daemon.start()
        except Exception as e:
            logger.critical(f"Daemon mode failed: {str(e)}", exc_info=True)
            sys.exit(1)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="DRN.today - Enterprise Lead Generation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes of operation:
  gui     - Launch graphical interface (default)
  cli     - Command-line interface
  headless- Headless validation (CI/CD)
  daemon  - Background service
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['gui', 'cli', 'headless', 'daemon'],
        default='gui',
        help='Execution mode (default: gui)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    # CLI-specific arguments
    subparsers = parser.add_subparsers(dest='command', help='CLI commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate system configuration')
    validate_parser.add_argument('--full', action='store_true', help='Full system validation')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('--format', choices=['csv', 'json', 'xlsx'], default='csv')
    export_parser.add_argument('--output', type=str, required=True)
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import data')
    import_parser.add_argument('--source', type=str, required=True)
    import_parser.add_argument('--type', choices=['leads', 'campaigns'], required=True)
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Create logs directory if it doesn't exist
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Parse arguments
    args = parse_arguments()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize application
    app = DRNMain()
    
    # Initialize core systems
    if not app.initialize_core_systems():
        logger.critical("Failed to initialize core systems")
        sys.exit(1)
    
    # Execute based on mode
    if args.mode == 'gui':
        app.run_gui_mode()
    elif args.mode == 'cli':
        app.run_cli_mode(args)
    elif args.mode == 'headless':
        success = app.run_headless_mode()
        sys.exit(0 if success else 1)
    elif args.mode == 'daemon':
        app.run_daemon_mode()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()