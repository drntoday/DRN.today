import sys
import os
import logging
import signal
from pathlib import Path
from typing import Optional

# Core engine imports
from engine.orchestrator import SystemOrchestrator
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager

# Configuration import
from home.config import get_config

# Initialize application logger
logger = logging.getLogger(__name__)

class DRNApplication:
    """Production-ready application initialization and lifecycle manager"""
    
    def __init__(self):
        self.config = get_config()
        self.event_bus: Optional[EventBus] = None
        self.storage: Optional[SecureStorage] = None
        self.license_manager: Optional[LicenseManager] = None
        self.orchestrator: Optional[SystemOrchestrator] = None
        self._shutdown_initiated = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def initialize(self) -> bool:
        """Initialize all core systems with production-grade error handling"""
        try:
            logger.info("Initializing DRN.today application...")
            
            # 1. Verify system requirements
            if not self._verify_system_requirements():
                logger.critical("System requirements verification failed")
                return False
                
            # 2. Initialize secure storage
            self.storage = SecureStorage(
                db_path=self.config.database.path,
                encryption_key=self.config.security.encryption_algorithm
            )
            if not self.storage.initialize():
                logger.critical("Secure storage initialization failed")
                return False
                
            # 3. Initialize license manager
            self.license_manager = LicenseManager(
                license_path=self.config.security.license_key_path,
                keyring_service=self.config.security.keyring_service
            )
            if not self.license_manager.validate():
                logger.critical("License validation failed")
                return False
                
            # 4. Initialize event system
            self.event_bus = EventBus()
            if not self.event_bus.start():
                logger.critical("Event system initialization failed")
                return False
                
            # 5. Initialize System orchestrator
            self.orchestrator = SystemOrchestrator(
                event_bus=self.event_bus,
                storage=self.storage,
                license_manager=self.license_manager
            )
            
            # 6. Load and initialize all modules
            if not self.orchestrator.load_modules():
                logger.critical("Module loading failed")
                return False
                
            if not self.orchestrator.initialize_modules():
                logger.critical("Module initialization failed")
                return False
                
            # 7. Start background services
            self._start_background_services()
            
            logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Application initialization failed: {str(e)}", exc_info=True)
            return False
    
    def _verify_system_requirements(self) -> bool:
        """Verify system meets minimum requirements"""
        try:
            # Check Python version
            if sys.version_info < (3, 9):
                logger.critical("Python 3.9 or higher is required")
                return False
                
            # Check available memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                if mem.available < 4 * 1024 * 1024 * 1024:  # 4GB
                    logger.warning("Less than 4GB RAM available - performance may be affected")
            except ImportError:
                logger.warning("psutil not available - cannot check memory")
                
            # Check disk space
            data_dir = Path(self.config.database.path).parent
            if data_dir.exists():
                disk_usage = os.statvfs(str(data_dir))
                free_space = disk_usage.f_frsize * disk_usage.f_bavail
                if free_space < 2 * 1024 * 1024 * 1024:  # 2GB
                    logger.warning("Less than 2GB disk space available")
                    
            # Verify AI model paths exist
            for model_name, model_path in self.config.get_ai_model_paths().items():
                if not Path(model_path).exists():
                    logger.error(f"AI model path not found: {model_path}")
                    return False
                    
            logger.info("System requirements verification passed")
            return True
            
        except Exception as e:
            logger.error(f"System requirements verification failed: {str(e)}")
            return False
    
    def _start_background_services(self):
        """Start essential background services"""
        try:
            # Start license monitoring
            self.license_manager.start_monitoring()
            
            # Start storage maintenance
            self.storage.start_maintenance(
                backup_interval=self.config.database.backup_interval_hours,
                max_connections=self.config.database.max_connections
            )
            
            # Start event system monitoring
            self.event_bus.start_monitoring()
            
            logger.info("Background services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start background services: {str(e)}")
            raise
    
    def run(self) -> int:
        """Main application execution loop"""
        if not self.initialize():
            return 1
            
        try:
            logger.info("DRN.today application started successfully")
            
            # Keep application running
            while not self._shutdown_initiated:
                try:
                    # Process events
                    self.event_bus.process_events()
                    
                    # Perform periodic maintenance
                    self.orchestrator.maintenance_cycle()
                    
                    # Small delay to prevent CPU overuse
                    import time
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
                    self.shutdown()
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                    
            return 0
            
        except Exception as e:
            logger.critical(f"Application runtime error: {str(e)}", exc_info=True)
            return 1
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of all systems"""
        if self._shutdown_initiated:
            return
            
        logger.info("Initiating application shutdown...")
        self._shutdown_initiated = True
        
        try:
            # Shutdown modules in reverse order
            if self.orchestrator:
                self.orchestrator.shutdown_modules()
                
            # Stop background services
            if self.license_manager:
                self.license_manager.stop_monitoring()
                
            if self.storage:
                self.storage.stop_maintenance()
                
            # Stop event system
            if self.event_bus:
                self.event_bus.stop()
                
            # Close storage
            if self.storage:
                self.storage.close()
                
            logger.info("Application shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.shutdown()
        sys.exit(0)

def create_application() -> DRNApplication:
    """Factory function to create and configure the application"""
    # Set up logging based on configuration
    logging.basicConfig(
        level=getattr(logging, os.getenv("DRN_LOG_LEVEL", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('drn.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and return application instance
    return DRNApplication()

def main():
    """Application entry point"""
    app = create_application()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
