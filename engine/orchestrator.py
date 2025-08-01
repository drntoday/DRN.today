import os
import sys
import importlib
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Core system imports
from engine.event_system import EventBus
from engine.storage import SecureStorage
from engine.license import LicenseManager
from home.config import get_config

# Initialize orchestrator logger
logger = logging.getLogger(__name__)

class BaseModule:
    """Base class for all feature modules"""
    
    def __init__(self, name: str, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager, config: Any):
        self.name = name
        self.event_bus = event_bus
        self.storage = storage
        self.license_manager = license_manager
        self.config = config
        self.initialized = False
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize the module with production-grade error handling"""
        try:
            logger.info(f"Initializing module: {self.name}")
            self._setup_event_handlers()
            self._validate_requirements()
            self.initialized = True
            logger.info(f"Module {self.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize module {self.name}: {str(e)}", exc_info=True)
            return False
    
    def start(self) -> bool:
        """Start the module's main functionality"""
        try:
            if not self.initialized:
                logger.error(f"Cannot start uninitialized module: {self.name}")
                return False
                
            logger.info(f"Starting module: {self.name}")
            self.running = True
            self._start_services()
            logger.info(f"Module {self.name} started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start module {self.name}: {str(e)}", exc_info=True)
            return False
    
    def stop(self) -> bool:
        """Stop the module's functionality gracefully"""
        try:
            if not self.running:
                return True
                
            logger.info(f"Stopping module: {self.name}")
            self._stop_services()
            self.running = False
            logger.info(f"Module {self.name} stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to stop module {self.name}: {str(e)}", exc_info=True)
            return False
    
    def maintenance_cycle(self):
        """Perform periodic maintenance tasks"""
        try:
            if self.running:
                self._perform_maintenance()
        except Exception as e:
            logger.error(f"Maintenance error in module {self.name}: {str(e)}", exc_info=True)
    
    def _setup_event_handlers(self):
        """Setup event handlers for the module"""
        pass
    
    def _validate_requirements(self):
        """Validate module requirements"""
        pass
    
    def _start_services(self):
        """Start module-specific services"""
        pass
    
    def _stop_services(self):
        """Stop module-specific services"""
        pass
    
    def _perform_maintenance(self):
        """Perform module-specific maintenance"""
        pass

class SystemOrchestrator:
    """Production-ready module orchestration system"""
    
    def __init__(self, event_bus: EventBus, storage: SecureStorage, 
                 license_manager: LicenseManager):
        self.event_bus = event_bus
        self.storage = storage
        self.license_manager = license_manager
        self.config = get_config()
        
        self.modules: Dict[str, BaseModule] = {}
        self.module_order: List[str] = [
            "compliance",
            "lead_generation",
            "lead_enrichment",
            "conversation_mining",
            "web_crawlers",
            "lead_capture",
            "competitive_intel",
            "email_system",
            "intent_engine",
            "template_engine",
            "integrations",
            "marketplace"
        ]
        
        self._module_paths: Dict[str, str] = {}
        self._module_classes: Dict[str, Type[BaseModule]] = {}
        
        # Setup event handlers
        self._setup_orchestrator_events()
    
    def _setup_orchestrator_events(self):
        """Setup orchestrator-level event handlers"""
        self.event_bus.subscribe("module.request", self._handle_module_request)
        self.event_bus.subscribe("system.shutdown", self._handle_system_shutdown)
    
    def load_modules(self) -> bool:
        """Dynamically load all feature modules with production error handling"""
        try:
            logger.info("Loading feature modules...")
            
            # Discover modules in the modules directory
            modules_dir = Path(__file__).parent.parent / "modules"
            if not modules_dir.exists():
                logger.error(f"Modules directory not found: {modules_dir}")
                return False
            
            # Load each module
            for module_name in self.module_order:
                module_dir = modules_dir / module_name
                if not module_dir.exists():
                    logger.warning(f"Module directory not found: {module_dir}")
                    continue
                
                # Find the main module file
                module_file = module_dir / f"{module_name}.py"
                if not module_file.exists():
                    logger.warning(f"Module file not found: {module_file}")
                    continue
                
                try:
                    # Import the module
                    module_path = f"modules.{module_name}.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Get the main class (should be named after the module in CamelCase)
                    class_name = "".join(word.capitalize() for word in module_name.split("_"))
                    if not hasattr(module, class_name):
                        logger.error(f"Module class {class_name} not found in {module_path}")
                        continue
                    
                    module_class = getattr(module, class_name)
                    if not issubclass(module_class, BaseModule):
                        logger.error(f"Module class {class_name} does not inherit from BaseModule")
                        continue
                    
                    # Store module info
                    self._module_paths[module_name] = module_path
                    self._module_classes[module_name] = module_class
                    logger.info(f"Discovered module: {module_name} -> {class_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load module {module_name}: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"Successfully loaded {len(self._module_classes)} modules")
            return True
            
        except Exception as e:
            logger.critical(f"Module loading failed: {str(e)}", exc_info=True)
            return False
    
    def initialize_modules(self) -> bool:
        """Initialize all loaded modules in dependency order"""
        try:
            logger.info("Initializing modules...")
            
            for module_name in self.module_order:
                if module_name not in self._module_classes:
                    continue
                
                # Check license permissions
                if not self.license_manager.has_module_access(module_name):
                    logger.warning(f"License does not permit access to module: {module_name}")
                    continue
                
                # Get module config
                module_config = self.config.get_module_config(module_name)
                
                # Instantiate module
                module_class = self._module_classes[module_name]
                module_instance = module_class(
                    name=module_name,
                    event_bus=self.event_bus,
                    storage=self.storage,
                    license_manager=self.license_manager,
                    config=module_config
                )
                
                # Initialize module
                if not module_instance.initialize():
                    logger.error(f"Failed to initialize module: {module_name}")
                    continue
                
                # Store initialized module
                self.modules[module_name] = module_instance
                logger.info(f"Module initialized: {module_name}")
            
            logger.info(f"Successfully initialized {len(self.modules)} modules")
            return True
            
        except Exception as e:
            logger.critical(f"Module initialization failed: {str(e)}", exc_info=True)
            return False
    
    def start_modules(self) -> bool:
        """Start all initialized modules in dependency order"""
        try:
            logger.info("Starting modules...")
            
            for module_name in self.module_order:
                if module_name not in self.modules:
                    continue
                
                module = self.modules[module_name]
                if not module.start():
                    logger.error(f"Failed to start module: {module_name}")
                    continue
                
                logger.info(f"Module started: {module_name}")
            
            logger.info("All modules started successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Module startup failed: {str(e)}", exc_info=True)
            return False
    
    def stop_modules(self) -> bool:
        """Stop all running modules in reverse dependency order"""
        try:
            logger.info("Stopping modules...")
            
            # Stop modules in reverse order
            for module_name in reversed(self.module_order):
                if module_name not in self.modules:
                    continue
                
                module = self.modules[module_name]
                if not module.stop():
                    logger.error(f"Failed to stop module: {module_name}")
                    continue
                
                logger.info(f"Module stopped: {module_name}")
            
            logger.info("All modules stopped successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Module shutdown failed: {str(e)}", exc_info=True)
            return False
    
    def maintenance_cycle(self):
        """Perform maintenance for all running modules"""
        try:
            for module_name, module in self.modules.items():
                if module.running:
                    module.maintenance_cycle()
        except Exception as e:
            logger.error(f"Module maintenance cycle failed: {str(e)}", exc_info=True)
    
    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """Get a module instance by name"""
        return self.modules.get(module_name)
    
    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        status = {}
        for module_name, module in self.modules.items():
            status[module_name] = {
                "initialized": module.initialized,
                "running": module.running,
                "has_license_access": self.license_manager.has_module_access(module_name)
            }
        return status
    
    def _handle_module_request(self, event_type: str, data: Dict[str, Any]):
        """Handle module requests via event system"""
        try:
            module_name = data.get("module")
            action = data.get("action")
            params = data.get("params", {})
            
            if not module_name or not action:
                logger.warning("Invalid module request: missing module or action")
                return
            
            module = self.get_module(module_name)
            if not module:
                logger.warning(f"Module not found: {module_name}")
                return
            
            # Execute action
            if hasattr(module, action):
                method = getattr(module, action)
                if callable(method):
                    result = method(**params)
                    self.event_bus.publish(f"module.{module_name}.{action}.response", {
                        "success": True,
                        "result": result
                    })
                else:
                    logger.warning(f"Module action not callable: {module_name}.{action}")
            else:
                logger.warning(f"Module action not found: {module_name}.{action}")
                
        except Exception as e:
            logger.error(f"Error handling module request: {str(e)}", exc_info=True)
            self.event_bus.publish("module.request.error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def _handle_system_shutdown(self, event_type: str, data: Dict[str, Any]):
        """Handle system shutdown event"""
        logger.info("Received system shutdown event")
        self.stop_modules()
    
    def shutdown_modules(self):
        """Shutdown all modules gracefully"""
        logger.info("Shutting down all modules...")
        self.stop_modules()
        self.modules.clear()
        logger.info("All modules shutdown complete")
