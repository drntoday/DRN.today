import os
import json
import logging
import locale
import ipaddress
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse

# Local imports matching project structure
from engine.storage import SecureStorage
from engine.event_system import EventBus, Event, EventPriority


class Region(Enum):
    """Supported regions for compliance"""
    EU = "EU"  # European Union (GDPR)
    US_CA = "US-CA"  # California (CCPA)
    US = "US"  # United States (Federal)
    UK = "UK"  # United Kingdom (UK GDPR)
    CA = "CA"  # Canada (PIPEDA)
    AU = "AU"  # Australia (Privacy Act)
    GLOBAL = "GLOBAL"  # Global default


class DataType(Enum):
    """Types of data that can be scraped"""
    PERSONAL = "personal"  # Personally identifiable information
    CONTACT = "contact"  # Contact information (email, phone)
    PROFESSIONAL = "professional"  # Professional information (job title, company)
    BEHAVIORAL = "behavioral"  # Behavioral data (clicks, engagement)
    PUBLIC = "public"  # Publicly available information


@dataclass
class DomainRestriction:
    """Restriction settings for a specific domain"""
    domain: str
    regions: List[Region]
    data_types: List[DataType]
    policy: str  # "allow", "block", "restrict"
    conditions: List[str]  # Conditions for restricted access
    last_updated: str  # ISO date string


@dataclass
class ScrapingDecision:
    """Decision result for a scraping request"""
    allowed: bool
    reason: str
    conditions: List[str]
    region: Region
    regulation: Optional[str] = None


class GeoRestrictions:
    """
    Geo-Aware Scraping Restrictions
    
    Implements geo-aware scraping restrictions to ensure compliance with
    regional privacy laws such as GDPR, CCPA, and other data protection regulations.
    """
    
    def __init__(self, storage: SecureStorage, event_system: EventBus):
        self.storage = storage
        self.event_system = event_system
        self.logger = logging.getLogger(__name__)
        
        # Default region (will be detected during initialization)
        self.current_region = Region.GLOBAL
        
        # Domain restrictions
        self.domain_restrictions: Dict[str, DomainRestriction] = {}
        
        # Global blacklisted domains
        self.global_blacklist: Set[str] = set()
        
        # Global whitelisted domains (always allowed)
        self.global_whitelist: Set[str] = set()
        
        # Region-specific regulations
        self.region_regulations = {
            Region.EU: "GDPR",
            Region.US_CA: "CCPA",
            Region.UK: "UK GDPR",
            Region.CA: "PIPEDA",
            Region.AU: "Privacy Act",
            Region.US: "Federal Privacy Laws",
            Region.GLOBAL: "Global Privacy Standards"
        }
        
        # Data type restrictions by region
        self.region_data_restrictions = {
            Region.EU: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "restrict",
                DataType.BEHAVIORAL: "block"
            },
            Region.US_CA: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "restrict",
                DataType.BEHAVIORAL: "restrict"
            },
            Region.UK: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "restrict",
                DataType.BEHAVIORAL: "block"
            },
            Region.CA: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "restrict",
                DataType.BEHAVIORAL: "restrict"
            },
            Region.AU: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "allow",
                DataType.BEHAVIORAL: "restrict"
            },
            Region.US: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "allow",
                DataType.BEHAVIORAL: "allow"
            },
            Region.GLOBAL: {
                DataType.PERSONAL: "restrict",
                DataType.CONTACT: "allow",
                DataType.BEHAVIORAL: "allow"
            }
        }
        
        # Load restrictions from storage
        self._load_restrictions()
        
        # Detect current region
        self._detect_region()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register event handlers for the event system"""
        self.event_system.subscribe(
            event_type="region_changed",
            handler=self._on_region_changed,
            priority=EventPriority.HIGH
        )
        
        self.event_system.subscribe(
            event_type="restrictions_updated",
            handler=self._on_restrictions_updated,
            priority=EventPriority.HIGH
        )
    
    def _load_restrictions(self):
        """Load restrictions from storage"""
        try:
            # Load domain restrictions
            restrictions_data = self.storage.load("compliance/domain_restrictions.json")
            if restrictions_data:
                for domain, restriction_data in restrictions_data.items():
                    # Convert regions and data types to enums
                    regions = [Region(r) for r in restriction_data.get("regions", [])]
                    data_types = [DataType(dt) for dt in restriction_data.get("data_types", [])]
                    
                    self.domain_restrictions[domain] = DomainRestriction(
                        domain=domain,
                        regions=regions,
                        data_types=data_types,
                        policy=restriction_data.get("policy", "allow"),
                        conditions=restriction_data.get("conditions", []),
                        last_updated=restriction_data.get("last_updated", "")
                    )
            
            # Load global blacklist
            blacklist_data = self.storage.load("compliance/global_blacklist.json")
            if blacklist_data:
                self.global_blacklist = set(blacklist_data.get("domains", []))
            
            # Load global whitelist
            whitelist_data = self.storage.load("compliance/global_whitelist.json")
            if whitelist_data:
                self.global_whitelist = set(whitelist_data.get("domains", []))
            
            self.logger.info(f"Loaded {len(self.domain_restrictions)} domain restrictions")
            self.logger.info(f"Loaded {len(self.global_blacklist)} blacklisted domains")
            self.logger.info(f"Loaded {len(self.global_whitelist)} whitelisted domains")
        except Exception as e:
            self.logger.error(f"Error loading restrictions: {str(e)}")
    
    def _detect_region(self):
        """Detect the current region based on system locale"""
        try:
            # Get system locale
            system_locale = locale.getdefaultlocale()[0]
            if not system_locale:
                self.logger.warning("Could not determine system locale, using global region")
                self.current_region = Region.GLOBAL
                return
            
            # Extract country code
            country_code = system_locale.split('_')[1] if '_' in system_locale else system_locale
            
            # Map country codes to regions
            country_to_region = {
                # EU countries
                'AT': Region.EU, 'BE': Region.EU, 'BG': Region.EU, 'HR': Region.EU,
                'CY': Region.EU, 'CZ': Region.EU, 'DK': Region.EU, 'EE': Region.EU,
                'FI': Region.EU, 'FR': Region.EU, 'DE': Region.EU, 'GR': Region.EU,
                'HU': Region.EU, 'IE': Region.EU, 'IT': Region.EU, 'LV': Region.EU,
                'LT': Region.EU, 'LU': Region.EU, 'MT': Region.EU, 'NL': Region.EU,
                'PL': Region.EU, 'PT': Region.EU, 'RO': Region.EU, 'SK': Region.EU,
                'SI': Region.EU, 'ES': Region.EU, 'SE': Region.EU,
                # UK
                'GB': Region.UK,
                # US
                'US': Region.US,
                # Canada
                'CA': Region.CA,
                # Australia
                'AU': Region.AU
            }
            
            # Special handling for US states
            if country_code == 'US':
                # Try to get more specific location if possible
                # In a real implementation, we might use IP geolocation or user settings
                # For now, we'll default to US federal
                self.current_region = Region.US
            else:
                self.current_region = country_to_region.get(country_code, Region.GLOBAL)
            
            self.logger.info(f"Detected region: {self.current_region.value}")
        except Exception as e:
            self.logger.error(f"Error detecting region: {str(e)}")
            self.current_region = Region.GLOBAL
    
    def set_region(self, region: Region):
        """
        Set the current region manually
        
        Args:
            region: Region to set
        """
        if region != self.current_region:
            old_region = self.current_region
            self.current_region = region
            self.logger.info(f"Region changed from {old_region.value} to {region.value}")
            
            # Emit event
            self.event_system.publish(Event(
                type="region_changed",
                data={
                    "old_region": old_region.value,
                    "new_region": region.value
                },
                priority=EventPriority.HIGH
            ))
    
    def add_domain_restriction(self, restriction: DomainRestriction):
        """
        Add or update a domain restriction
        
        Args:
            restriction: Domain restriction to add
        """
        self.domain_restrictions[restriction.domain] = restriction
        self._save_domain_restrictions()
        
        # Emit event
        self.event_system.publish(Event(
            type="restrictions_updated",
            data={
                "action": "add",
                "domain": restriction.domain,
                "policy": restriction.policy
            },
            priority=EventPriority.HIGH
        ))
        
        self.logger.info(f"Added restriction for domain: {restriction.domain}")
    
    def remove_domain_restriction(self, domain: str):
        """
        Remove a domain restriction
        
        Args:
            domain: Domain to remove restriction for
        """
        if domain in self.domain_restrictions:
            del self.domain_restrictions[domain]
            self._save_domain_restrictions()
            
            # Emit event
            self.event_system.publish(Event(
                type="restrictions_updated",
                data={
                    "action": "remove",
                    "domain": domain
                },
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Removed restriction for domain: {domain}")
    
    def add_to_blacklist(self, domain: str):
        """
        Add a domain to the global blacklist
        
        Args:
            domain: Domain to blacklist
        """
        self.global_blacklist.add(domain)
        self._save_blacklist()
        
        # Emit event
        self.event_system.publish(Event(
            type="restrictions_updated",
            data={
                "action": "blacklist",
                "domain": domain
            },
            priority=EventPriority.HIGH
        ))
        
        self.logger.info(f"Added domain to blacklist: {domain}")
    
    def remove_from_blacklist(self, domain: str):
        """
        Remove a domain from the global blacklist
        
        Args:
            domain: Domain to remove from blacklist
        """
        if domain in self.global_blacklist:
            self.global_blacklist.remove(domain)
            self._save_blacklist()
            
            # Emit event
            self.event_system.publish(Event(
                type="restrictions_updated",
                data={
                    "action": "unblacklist",
                    "domain": domain
                },
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Removed domain from blacklist: {domain}")
    
    def add_to_whitelist(self, domain: str):
        """
        Add a domain to the global whitelist
        
        Args:
            domain: Domain to whitelist
        """
        self.global_whitelist.add(domain)
        self._save_whitelist()
        
        # Emit event
        self.event_system.publish(Event(
            type="restrictions_updated",
            data={
                "action": "whitelist",
                "domain": domain
            },
            priority=EventPriority.HIGH
        ))
        
        self.logger.info(f"Added domain to whitelist: {domain}")
    
    def remove_from_whitelist(self, domain: str):
        """
        Remove a domain from the global whitelist
        
        Args:
            domain: Domain to remove from whitelist
        """
        if domain in self.global_whitelist:
            self.global_whitelist.remove(domain)
            self._save_whitelist()
            
            # Emit event
            self.event_system.publish(Event(
                type="restrictions_updated",
                data={
                    "action": "unwhitelist",
                    "domain": domain
                },
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Removed domain from whitelist: {domain}")
    
    def is_scraping_allowed(self, domain: str, data_type: DataType = None) -> ScrapingDecision:
        """
        Check if scraping is allowed for a domain
        
        Args:
            domain: Domain to check
            data_type: Type of data to scrape
            
        Returns:
            ScrapingDecision: Decision result
        """
        try:
            # Normalize domain
            domain = self._normalize_domain(domain)
            
            # Check global whitelist first
            if domain in self.global_whitelist:
                return ScrapingDecision(
                    allowed=True,
                    reason="Domain is in global whitelist",
                    conditions=[],
                    region=self.current_region
                )
            
            # Check global blacklist
            if domain in self.global_blacklist:
                return ScrapingDecision(
                    allowed=False,
                    reason="Domain is in global blacklist",
                    conditions=[],
                    region=self.current_region
                )
            
            # Check domain-specific restrictions
            if domain in self.domain_restrictions:
                restriction = self.domain_restrictions[domain]
                
                # Check if current region is in the restriction's regions
                if self.current_region in restriction.regions or Region.GLOBAL in restriction.regions:
                    # Check data type restrictions
                    if data_type and data_type in restriction.data_types:
                        if restriction.policy == "block":
                            return ScrapingDecision(
                                allowed=False,
                                reason=f"Scraping {data_type.value} data is blocked for this domain in {self.current_region.value}",
                                conditions=[],
                                region=self.current_region,
                                regulation=self.region_regulations.get(self.current_region)
                            )
                        elif restriction.policy == "restrict":
                            return ScrapingDecision(
                                allowed=True,
                                reason=f"Scraping {data_type.value} data is restricted for this domain in {self.current_region.value}",
                                conditions=restriction.conditions,
                                region=self.current_region,
                                regulation=self.region_regulations.get(self.current_region)
                            )
                    
                    # If no data type restriction or data type not restricted, check general policy
                    if restriction.policy == "block":
                        return ScrapingDecision(
                            allowed=False,
                            reason=f"Scraping is blocked for this domain in {self.current_region.value}",
                            conditions=[],
                            region=self.current_region,
                            regulation=self.region_regulations.get(self.current_region)
                        )
                    elif restriction.policy == "restrict":
                        return ScrapingDecision(
                            allowed=True,
                            reason=f"Scraping is restricted for this domain in {self.current_region.value}",
                            conditions=restriction.conditions,
                            region=self.current_region,
                            regulation=self.region_regulations.get(self.current_region)
                        )
            
            # Check region-level data type restrictions
            if data_type:
                region_restrictions = self.region_data_restrictions.get(self.current_region, {})
                if data_type in region_restrictions:
                    policy = region_restrictions[data_type]
                    if policy == "block":
                        return ScrapingDecision(
                            allowed=False,
                            reason=f"Scraping {data_type.value} data is blocked in {self.current_region.value}",
                            conditions=[],
                            region=self.current_region,
                            regulation=self.region_regulations.get(self.current_region)
                        )
                    elif policy == "restrict":
                        return ScrapingDecision(
                            allowed=True,
                            reason=f"Scraping {data_type.value} data is restricted in {self.current_region.value}",
                            conditions=["consent", "transparency"],
                            region=self.current_region,
                            regulation=self.region_regulations.get(self.current_region)
                        )
            
            # Default: allow
            return ScrapingDecision(
                allowed=True,
                reason="No restrictions apply",
                conditions=[],
                region=self.current_region
            )
        except Exception as e:
            self.logger.error(f"Error checking scraping restrictions for {domain}: {str(e)}")
            # Default to allow in case of error to avoid blocking legitimate requests
            return ScrapingDecision(
                allowed=True,
                reason="Error checking restrictions, allowing by default",
                conditions=[],
                region=self.current_region
            )
    
    def _normalize_domain(self, domain: str) -> str:
        """
        Normalize a domain name
        
        Args:
            domain: Domain to normalize
            
        Returns:
            str: Normalized domain
        """
        try:
            # Parse URL if it's a full URL
            if "://" in domain:
                parsed = urlparse(domain)
                domain = parsed.netloc
            
            # Remove port if present
            if ":" in domain:
                domain = domain.split(":")[0]
            
            # Convert to lowercase
            domain = domain.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            return domain
        except Exception as e:
            self.logger.error(f"Error normalizing domain {domain}: {str(e)}")
            return domain
    
    def _save_domain_restrictions(self):
        """Save domain restrictions to storage"""
        try:
            restrictions_data = {}
            for domain, restriction in self.domain_restrictions.items():
                restrictions_data[domain] = {
                    "regions": [r.value for r in restriction.regions],
                    "data_types": [dt.value for dt in restriction.data_types],
                    "policy": restriction.policy,
                    "conditions": restriction.conditions,
                    "last_updated": restriction.last_updated
                }
            
            self.storage.save("compliance/domain_restrictions.json", restrictions_data)
        except Exception as e:
            self.logger.error(f"Error saving domain restrictions: {str(e)}")
    
    def _save_blacklist(self):
        """Save global blacklist to storage"""
        try:
            blacklist_data = {"domains": list(self.global_blacklist)}
            self.storage.save("compliance/global_blacklist.json", blacklist_data)
        except Exception as e:
            self.logger.error(f"Error saving blacklist: {str(e)}")
    
    def _save_whitelist(self):
        """Save global whitelist to storage"""
        try:
            whitelist_data = {"domains": list(self.global_whitelist)}
            self.storage.save("compliance/global_whitelist.json", whitelist_data)
        except Exception as e:
            self.logger.error(f"Error saving whitelist: {str(e)}")
    
    def _on_region_changed(self, event: Event):
        """Handle region changed event"""
        old_region = event.data.get("old_region")
        new_region = event.data.get("new_region")
        self.logger.info(f"Region changed from {old_region} to {new_region}")
    
    def _on_restrictions_updated(self, event: Event):
        """Handle restrictions updated event"""
        action = event.data.get("action")
        domain = event.data.get("domain")
        self.logger.info(f"Restrictions updated: {action} {domain}")
    
    def get_domain_restrictions(self, domain: str) -> Optional[DomainRestriction]:
        """
        Get restrictions for a specific domain
        
        Args:
            domain: Domain to get restrictions for
            
        Returns:
            DomainRestriction: Domain restrictions or None if not found
        """
        domain = self._normalize_domain(domain)
        return self.domain_restrictions.get(domain)
    
    def get_all_domain_restrictions(self) -> Dict[str, DomainRestriction]:
        """
        Get all domain restrictions
        
        Returns:
            Dict[str, DomainRestriction]: All domain restrictions
        """
        return self.domain_restrictions.copy()
    
    def get_blacklist(self) -> Set[str]:
        """
        Get the global blacklist
        
        Returns:
            Set[str]: Blacklisted domains
        """
        return self.global_blacklist.copy()
    
    def get_whitelist(self) -> Set[str]:
        """
        Get the global whitelist
        
        Returns:
            Set[str]: Whitelisted domains
        """
        return self.global_whitelist.copy()
    
    def get_current_region(self) -> Region:
        """
        Get the current region
        
        Returns:
            Region: Current region
        """
        return self.current_region
    
    def get_region_regulation(self, region: Region = None) -> Optional[str]:
        """
        Get the regulation for a region
        
        Args:
            region: Region to get regulation for (defaults to current region)
            
        Returns:
            str: Regulation name or None if not found
        """
        if region is None:
            region = self.current_region
        return self.region_regulations.get(region)
    
    def get_region_data_restrictions(self, region: Region = None) -> Dict[DataType, str]:
        """
        Get data type restrictions for a region
        
        Args:
            region: Region to get restrictions for (defaults to current region)
            
        Returns:
            Dict[DataType, str]: Data type restrictions
        """
        if region is None:
            region = self.current_region
        return self.region_data_restrictions.get(region, {}).copy()
    
    def is_ip_address_allowed(self, ip_address: str) -> bool:
        """
        Check if an IP address is allowed based on geo restrictions
        
        Args:
            ip_address: IP address to check
            
        Returns:
            bool: True if allowed, False otherwise
        """
        try:
            # Parse IP address
            ip = ipaddress.ip_address(ip_address)
            
            # Check if IP is in a restricted range
            # This is a simplified implementation - in a real-world scenario,
            # you would use a GeoIP database to determine the country of the IP
            
            # For now, we'll just check if it's a private IP
            if ip.is_private:
                return True
            
            # In a real implementation, you would use a GeoIP database to get the country
            # and then check if that country has restrictions
            
            # Default to allow
            return True
        except Exception as e:
            self.logger.error(f"Error checking IP address {ip_address}: {str(e)}")
            return True
    
    def export_restrictions(self, file_path: str):
        """
        Export restrictions to a file
        
        Args:
            file_path: Path to export to
        """
        try:
            export_data = {
                "current_region": self.current_region.value,
                "domain_restrictions": {
                    domain: {
                        "regions": [r.value for r in restriction.regions],
                        "data_types": [dt.value for dt in restriction.data_types],
                        "policy": restriction.policy,
                        "conditions": restriction.conditions,
                        "last_updated": restriction.last_updated
                    }
                    for domain, restriction in self.domain_restrictions.items()
                },
                "global_blacklist": list(self.global_blacklist),
                "global_whitelist": list(self.global_whitelist),
                "region_regulations": {
                    region.value: regulation for region, regulation in self.region_regulations.items()
                },
                "region_data_restrictions": {
                    region.value: {
                        data_type.value: policy for data_type, policy in restrictions.items()
                    }
                    for region, restrictions in self.region_data_restrictions.items()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported restrictions to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting restrictions: {str(e)}")
    
    def import_restrictions(self, file_path: str):
        """
        Import restrictions from a file
        
        Args:
            file_path: Path to import from
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            # Import current region
            if "current_region" in import_data:
                try:
                    region = Region(import_data["current_region"])
                    self.set_region(region)
                except ValueError:
                    self.logger.warning(f"Invalid region in import file: {import_data['current_region']}")
            
            # Import domain restrictions
            if "domain_restrictions" in import_data:
                for domain, restriction_data in import_data["domain_restrictions"].items():
                    try:
                        regions = [Region(r) for r in restriction_data.get("regions", [])]
                        data_types = [DataType(dt) for dt in restriction_data.get("data_types", [])]
                        
                        restriction = DomainRestriction(
                            domain=domain,
                            regions=regions,
                            data_types=data_types,
                            policy=restriction_data.get("policy", "allow"),
                            conditions=restriction_data.get("conditions", []),
                            last_updated=restriction_data.get("last_updated", "")
                        )
                        
                        self.domain_restrictions[domain] = restriction
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Invalid restriction data for domain {domain}: {str(e)}")
            
            # Import global blacklist
            if "global_blacklist" in import_data:
                self.global_blacklist = set(import_data["global_blacklist"])
            
            # Import global whitelist
            if "global_whitelist" in import_data:
                self.global_whitelist = set(import_data["global_whitelist"])
            
            # Import region regulations (optional)
            if "region_regulations" in import_data:
                for region_str, regulation in import_data["region_regulations"].items():
                    try:
                        region = Region(region_str)
                        self.region_regulations[region] = regulation
                    except ValueError:
                        self.logger.warning(f"Invalid region in regulations: {region_str}")
            
            # Import region data restrictions (optional)
            if "region_data_restrictions" in import_data:
                for region_str, restrictions in import_data["region_data_restrictions"].items():
                    try:
                        region = Region(region_str)
                        parsed_restrictions = {}
                        for data_type_str, policy in restrictions.items():
                            try:
                                data_type = DataType(data_type_str)
                                parsed_restrictions[data_type] = policy
                            except ValueError:
                                self.logger.warning(f"Invalid data type in restrictions: {data_type_str}")
                        self.region_data_restrictions[region] = parsed_restrictions
                    except ValueError:
                        self.logger.warning(f"Invalid region in data restrictions: {region_str}")
            
            # Save imported restrictions
            self._save_domain_restrictions()
            self._save_blacklist()
            self._save_whitelist()
            
            # Emit event
            self.event_system.publish(Event(
                type="restrictions_updated",
                data={
                    "action": "import",
                    "source": file_path
                },
                priority=EventPriority.HIGH
            ))
            
            self.logger.info(f"Imported restrictions from {file_path}")
        except Exception as e:
            self.logger.error(f"Error importing restrictions: {str(e)}")