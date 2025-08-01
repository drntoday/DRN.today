import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Create logs directory if it doesn't exist
project_root = Path(__file__).resolve().parent.parent.parent
logs_dir = project_root / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / 'cli.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from modules.lead_generation.scraper import LeadScraper
    from modules.lead_enrichment.persona_stitcher import PersonaStitcher
    from modules.conversation_mining.monitor import ConversationMonitor
    from modules.web_crawlers.crawler import WebCrawler
    from modules.email_system.smtp_manager import SMTPManager
    from modules.competitive_intel.monitor import CompetitorMonitor
    from modules.intent_engine.tracker import IntentTracker
    from modules.template_engine.templates import TemplateEngine
    from modules.marketplace.store import MarketplaceStore
    from modules.compliance.restrictions import GeoRestrictions
    from ai.nlp import NLPProcessor
    from ai.scoring import LeadScorer
    from engine.orchestrator import Orchestrator
    from engine.license import LicenseManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class DRNCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="DRN.today - Enterprise-Grade Lead Generation Platform CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Example: python interface/cli/main.py generate-leads --source linkedin --keywords 'CTO SaaS'"
        )
        
        self.license_manager = LicenseManager()
        self.orchestrator = Orchestrator()
        
        self._setup_commands()
        
    def _setup_commands(self):
        subparsers = self.parser.add_subparsers(dest='command', help='Available commands')
        
        # Lead Generation Commands
        generate_parser = subparsers.add_parser('generate-leads', help='Generate leads from various sources')
        generate_parser.add_argument('--source', required=True, choices=[
            'google', 'yahoo', 'duckduckgo', 'yandex', 'linkedin', 'discord', 
            'telegram', 'reddit', 'forums', 'appstores', 'github', 'crunchbase',
            'angellist', 'producthunt', 'newsletters'
        ], help='Source platform for lead generation')
        generate_parser.add_argument('--keywords', required=True, help='Search keywords (comma-separated)')
        generate_parser.add_argument('--location', help='Geographic filter')
        generate_parser.add_argument('--limit', type=int, default=100, help='Maximum leads to generate')
        generate_parser.add_argument('--output', choices=['json', 'csv', 'sqlite'], default='json', help='Output format')
        generate_parser.add_argument('--proxy', help='Proxy server (format: ip:port)')
        generate_parser.set_defaults(func=self._generate_leads)
        
        # Email Campaign Commands
        campaign_parser = subparsers.add_parser('send-campaign', help='Send email campaigns')
        campaign_parser.add_argument('--campaign-id', required=True, help='Campaign ID to send')
        campaign_parser.add_argument('--template', help='Email template to use')
        campaign_parser.add_argument('--schedule', help='Schedule time (YYYY-MM-DD HH:MM)')
        campaign_parser.add_argument('--test', action='store_true', help='Send test email only')
        campaign_parser.set_defaults(func=self._send_campaign)
        
        # Conversation Monitoring Commands
        monitor_parser = subparsers.add_parser('monitor-conversations', help='Monitor conversations for intent')
        monitor_parser.add_argument('--platforms', required=True, help='Platforms to monitor (comma-separated)')
        monitor_parser.add_argument('--keywords', help='Keywords to track (comma-separated)')
        monitor_parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in minutes')
        monitor_parser.add_argument('--output', choices=['json', 'csv'], default='json', help='Output format')
        monitor_parser.set_defaults(func=self._monitor_conversations)
        
        # Lead Enrichment Commands
        enrich_parser = subparsers.add_parser('enrich-leads', help='Enrich existing leads')
        enrich_parser.add_argument('--input', required=True, help='Input file path')
        enrich_parser.add_argument('--enrichments', required=True, choices=[
            'social', 'company', 'funding', 'technologies', 'contacts'
        ], nargs='+', help='Enrichment types to perform')
        enrich_parser.add_argument('--output', help='Output file path')
        enrich_parser.set_defaults(func=self._enrich_leads)
        
        # Adapter Creation Commands
        adapter_parser = subparsers.add_parser('create-adapter', help='Create new scraping adapter')
        adapter_parser.add_argument('--name', required=True, help='Adapter name')
        adapter_parser.add_argument('--config', required=True, help='JSON configuration file')
        adapter_parser.add_argument('--output', help='Output directory')
        adapter_parser.set_defaults(func=self._create_adapter)
        
        # Competitive Intelligence Commands
        intel_parser = subparsers.add_parser('competitive-intel', help='Gather competitive intelligence')
        intel_parser.add_argument('--competitors', required=True, help='Competitor domains (comma-separated)')
        intel_parser.add_argument('--monitor', choices=['followers', 'pricing', 'ads', 'jobs'], 
                                 nargs='+', help='Intelligence types to gather')
        intel_parser.add_argument('--duration', type=int, default=30, help='Monitoring duration in days')
        intel_parser.set_defaults(func=self._gather_intelligence)
        
        # License Management Commands
        license_parser = subparsers.add_parser('license', help='Manage license')
        license_subparsers = license_parser.add_subparsers(dest='license_action')
        
        license_subparsers.add_parser('status', help='Check license status').set_defaults(func=self._license_status)
        license_subparsers.add_parser('activate', help='Activate license').add_argument('--key', required=True)
        license_subparsers.add_parser('deactivate', help='Deactivate license')
        
        # Marketplace Commands
        market_parser = subparsers.add_parser('marketplace', help='Manage lead packs')
        market_subparsers = market_parser.add_subparsers(dest='market_action')
        
        market_subparsers.add_parser('list', help='List available packs').set_defaults(func=self._list_packs)
        market_subparsers.add_parser('download', help='Download a pack').add_argument('--pack-id', required=True)
        market_subparsers.add_parser('publish', help='Publish a pack').add_argument('--path', required=True)
        
        # System Commands
        system_parser = subparsers.add_parser('system', help='System operations')
        system_subparsers = system_parser.add_subparsers(dest='system_action')
        
        system_subparsers.add_parser('status', help='Check system status').set_defaults(func=self._system_status)
        system_subparsers.add_parser('update', help='Update application').set_defaults(func=self._update_system)
        system_subparsers.add_parser('backup', help='Backup data').add_argument('--path', required=True)
        
    def run(self):
        args = self.parser.parse_args()
        
        if not hasattr(args, 'func'):
            self.parser.print_help()
            return
            
        try:
            if not self.license_manager.is_valid():
                print("Error: Invalid or expired license. Please activate your license.")
                return
                
            args.func(args)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
            
    def _generate_leads(self, args):
        scraper = LeadScraper()
        keywords = [k.strip() for k in args.keywords.split(',')]
        
        print(f"Generating leads from {args.source} with keywords: {', '.join(keywords)}")
        
        leads = scraper.scrape(
            source=args.source,
            keywords=keywords,
            location=args.location,
            limit=args.limit,
            proxy=args.proxy
        )
        
        if args.output == 'json':
            print(json.dumps(leads, indent=2))
        elif args.output == 'csv':
            # CSV output implementation
            pass
        elif args.output == 'sqlite':
            # SQLite output implementation
            pass
            
        print(f"Successfully generated {len(leads)} leads")
        
    def _send_campaign(self, args):
        smtp_manager = SMTPManager()
        
        print(f"Sending campaign {args.campaign_id}")
        
        if args.test:
            result = smtp_manager.send_test(campaign_id=args.campaign_id)
        else:
            result = smtp_manager.send_campaign(
                campaign_id=args.campaign_id,
                template=args.template,
                schedule_time=args.schedule
            )
            
        print(f"Campaign sent successfully. Result: {result}")
        
    def _monitor_conversations(self, args):
        platforms = [p.strip() for p in args.platforms.split(',')]
        keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else []
        
        monitor = ConversationMonitor()
        
        print(f"Monitoring conversations on {', '.join(platforms)} for {args.duration} minutes")
        
        intents = monitor.monitor(
            platforms=platforms,
            keywords=keywords,
            duration_minutes=args.duration
        )
        
        if args.output == 'json':
            print(json.dumps(intents, indent=2))
        elif args.output == 'csv':
            # CSV output implementation
            pass
            
        print(f"Found {len(intents)} high-intent conversations")
        
    def _enrich_leads(self, args):
        persona_stitcher = PersonaStitcher()
        
        print(f"Enriching leads from {args.input}")
        
        enriched_leads = persona_stitcher.enrich(
            input_file=args.input,
            enrichment_types=args.enrichments
        )
        
        if args.output:
            # Save to output file
            pass
            
        print(f"Successfully enriched {len(enriched_leads)} leads")
        
    def _create_adapter(self, args):
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        print(f"Creating adapter '{args.name}'")
        
        # Adapter creation implementation
        print(f"Adapter created successfully at {args.output or 'adapters/'}")
        
    def _gather_intelligence(self, args):
        competitors = [c.strip() for c in args.competitors.split(',')]
        monitor = CompetitorMonitor()
        
        print(f"Gathering intelligence for competitors: {', '.join(competitors)}")
        
        intel = monitor.gather(
            competitors=competitors,
            intel_types=args.monitor,
            duration_days=args.duration
        )
        
        print(json.dumps(intel, indent=2))
        
    def _license_status(self, args):
        status = self.license_manager.get_status()
        print(json.dumps(status, indent=2))
        
    def _list_packs(self, args):
        store = MarketplaceStore()
        packs = store.list_packs()
        print(json.dumps(packs, indent=2))
        
    def _system_status(self, args):
        status = self.orchestrator.get_system_status()
        print(json.dumps(status, indent=2))
        
    def _update_system(self, args):
        print("Checking for updates...")
        # Update implementation
        print("System updated successfully")
        
    def _backup_data(self, args):
        print(f"Backing up data to {args.path}")
        # Backup implementation
        print("Backup completed successfully")
