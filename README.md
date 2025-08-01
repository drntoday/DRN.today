DRN.today - Enterprise-Grade Lead Generation Platform
DRN.today Logo

DRN.today is a fully-featured, enterprise-grade desktop application designed for automated lead generation and outreach. It runs seamlessly across Windows, Linux, and macOS with all processing occurring locally on the user's machine for maximum privacy and security.

Table of Contents
Features
Technology Stack
Installation
Project Structure
Usage
Licensing & Pricing
Compliance & Ethics
Contributing
Support
Features
🔥 Unrivaled AI-Powered Lead Generation Engine
Multi-platform scraping from Google, Yahoo, DuckDuckGo, Yandex, LinkedIn, Discord, Telegram, Reddit, forums, app stores, GitHub, Crunchbase, AngelList, Product Hunt, newsletters, and more
WHOIS & DNS-based email mining, startup funding monitors, newsletter author crawlers, and Chrome extension data capture
Dynamic adapter architecture for pluggable sources, powered by Playwright with TinyBERT-based DOM self-healing and adapter chaining
Multi-hop crawling with smart retries, proxy support, CAPTCHA solving with OpenCV, and failure auto-patching
Language-agnostic smart extraction of names, emails, phones, companies, social links, locations, and contextual keywords
Built-in TinyBERT NLP for scoring, category labeling, relevance filtering, and buyer persona detection
GUI-based Adapter Builder to convert custom JSON rules into working Python crawlers without code
🧠 Deep Lead Enrichment & AI-Based Persona Stitching
Cross-platform identity resolution using domain, name, role, and social footprint
AI-inferred insights such as job seniority, budget range, urgency, industry fit, and authority level using TinyBERT
DNA-style lead tagging system for targeting leads by multi-dimensional traits
Custom persona segmentation (e.g., "CTOs in FinTech startups with Series A funding")
💬 High-Intent Conversation Mining
Real-time monitoring of forums, Discord, Reddit, Telegram, and comment sections
NLP-driven detection of buying signals (e.g., "looking for", "need service", "recommend a tool") using TinyBERT
Thread classification, contextual extraction, and cluster scoring based on lead urgency
🛠️ Self-Adaptive, Multi-Stage Web Crawlers
Fully autonomous crawlers that self-heal using TinyBERT DOM-mapping if selectors break
Multi-layered crawling (source > result > landing > social > enrichment)
Smart retry logic, user-agent spoofing, and rate-limit awareness built-in
📩 Passive Lead Capture & Lead Magnet Generators
Auto-generated SEO blog network publishing niche content with lead CTAs using GPT-2
Public tool repositories, GitHub projects, and gated downloads to capture leads
Deployable Chrome extensions that log anonymous leads and usage-based context
📊 Competitive Intelligence Engine
Monitor and scrape competitor followers across social platforms
Track pricing, landing pages, and job board changes
Watch Google Ads/PPC campaigns and extract contact points from sponsored links
🌐 Truly Universal Email & SMTP Compatibility
Full SMTP integration with Gmail, Outlook, Mailgun, SendGrid, Amazon SES, Postfix, and more
Secure keyring-based credential storage and SMTP pool rotation logic
Real-time blacklist monitoring and warming automation
📬 Smart IMAP + Bounce Management
Works with any IMAP provider
ML-based reply classification (positive, neutral, spam, bounced) using Scikit-learn and TinyBERT
Bounce removal and automatic lead rescore system
💡 Live Intent Engine & Heatmaps
Tracks opens, clicks, re-engagement, scroll depth, and dwell time
AI recommends best next outreach strategy based on behavioral signals using TinyBERT
Generates visual heatmaps per industry and per campaign
📝 AI-Driven Email Template Engine
Jinja2-powered, context-aware templates adapting tone and structure to each lead
Website-aware personalization, follow-up sequences, and smart merge logic
A/B testing support with conditional branching and lead flow tracking
🔌 Integrations via Webhooks & REST API
One-click sync with Notion, Airtable, Slack, CRMs, Zapier, and custom tools
Data exports supported in CSV, Excel, JSON, SQLite, and live Webhook streaming
Local webhook server for real-time integrations without external dependencies
🖥️ Multi-Mode Operation
PyQt5 GUI interface for full feature control
Command-line batch runner for headless servers
Background daemon with watchdogs and dynamic scheduling
🔒 Licensing, Subscriptions & Access Control
Offline-ready license key validation with expiration and role-based features
Role tiers: Admin, Standard, and Trial
Optional billing integration with Stripe or Paddle
🚀 True Cross-Platform Packaging & Deployment
Full bundling for Windows (.exe), macOS (.app), and Linux (.AppImage, .deb)
PyInstaller, Briefcase, or Nuitka for one-file or directory-based deployment
Includes all internal resources: plugins, templates, icons, and pretrained TinyBERT models
🧠 AI Lead Scoring Engine
ML models continuously evaluate leads based on 20+ signals using Scikit-learn and TinyBERT
Predicts conversion probability and campaign responsiveness
Allows automatic filtering or retargeting based on predicted value
🛒 Lead Pack Marketplace
Built-in store for downloading or creating lead generation packs
Industry-targeted scraping adapters, ready-to-run campaign blueprints, and enrichment logic
Enables new monetization or sharing models via community contributions
🛡️ GDPR, CCPA, and Data Ethics Support
Geo-aware scraping restrictions
Built-in opt-out flows and data retention rules
Source-level blacklisting and scraping compliance logic
Technology Stack
Core Technologies
Python 3.9+: Primary programming language
PyQt5: Cross-platform GUI framework
SQLite: Local data storage
Playwright: Browser automation for web scraping
TinyBERT: Lightweight NLP model for local AI processing
Scikit-learn: Machine learning library for lead scoring
Jinja2: Template engine for email personalization
AI & ML
TinyBERT: For NLP tasks including entity recognition, sentiment analysis, and intent detection
Scikit-learn: For lead scoring and classification
OpenCV: For CAPTCHA solving
GPT-2: For content generation
Data Processing
Pandas: Data manipulation and analysis
NumPy: Numerical computing
BeautifulSoup: HTML parsing
LXML: XML and HTML processing
Networking & APIs
Requests: HTTP library
AIOHTTP: Async HTTP client/server
SMTPlib & IMAPlib: Email communication
Flask: For local webhook server
Packaging & Deployment
PyInstaller: For creating standalone executables
Briefcase: For cross-platform packaging
Nuitka: Python compiler for optimized executables
Installation
Prerequisites
Python 3.9 or higher
pip package manager
4GB+ RAM (8GB recommended)
2GB+ free disk space

Step 1: Clone the Repository
git clone https://github.com/yourusername/DRN.today.git
cd DRN.today

Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Download AI Models
python scripts/download_models.py

Step 5: Run the Application
python home/main.py


Project Structure

DRN.today/
├── README.md
├── requirements.txt
│ 
├── .github/                    
│   └── workflows/      
│        └── ci.yml
│            
├── .gitignore
│
├── 🏠 home/                      # Core application entry point
│   ├── main.py                   # Application orchestrator 
│   ├── app.py                    # Application initialization
│   └── config.py                 # Global configuration
│
├── 🔧 engine/                    # Core processing engine
│   ├── orchestrator.py           # Central module coordinator
│   ├── event_system.py           # Inter-module communication
│   ├── storage.py                # Local data persistence
│   └── license.py                # Licensing & access control
│
├── 🧩 modules/                   # Feature modules 
│   ├── 🎯 lead_generation/       # Lead generation features
│   │   ├── scraper.py            # Multi-platform scraping
│   │   ├── adapter_builder.py    # GUI adapter builder
│   │   └── email_miner.py        # WHOIS & DNS mining
│   │
│   ├── 💎 lead_enrichment/       # Lead enrichment features
│   │   ├── persona_stitcher.py   # Cross-platform identity resolution
│   │   ├── tagging.py            # DNA-style lead tagging
│   │   └── insights.py           # AI-inferred insights
│   │
│   ├── 💬 conversation_mining/   # Conversation mining features
│   │   ├── monitor.py            # Real-time monitoring
│   │   ├── intent_detector.py    # NLP buying signal detection
│   │   │
│   │   ├── default_training_examples.py
│   │   │
│   │   └── classifier.py         # Thread classification
│   │
│   ├── 🕷️ web_crawlers/          # Self-adaptive crawlers
│   │   ├── crawler.py            # Autonomous crawler
│   │   ├── self_healing.py       # TinyBERT DOM-mapping
│   │   └── retry_logic.py        # Smart retry system
│   │
│   ├── 📩 lead_capture/          # Passive lead capture
│   │   ├── content_generator.py  # SEO blog network
│   │   ├── tools.py              # Public tool repositories
│   │   └── extension.py          # Chrome extension
│   │
│   ├── 🔍 competitive_intel/     # Competitive intelligence
│   │   ├── monitor.py            # Competitor tracking
│   │   ├── scraper.py            # Pricing & landing page scraping
│   │   └── ad_tracker.py         # Google Ads monitoring
│   │
│   ├── 📧 email_system/          # Email & SMTP system
│   │   ├── smtp_manager.py       # SMTP integration
│   │   ├── imap_manager.py       # IMAP processing
│   │   └── bounce_detector.py    # Bounce detection
│   │
│   ├── 📈 intent_engine/         # Live intent tracking
│   │   ├── tracker.py            # Behavior tracking
│   │   ├── heatmap.py            # Visual heatmaps
│   │   └── recommender.py        # AI recommendations
│   │
│   ├── 📝 template_engine/       # Email template system
│   │   ├── templates.py          # Jinja2 template engine
│   │   ├── personalizer.py       # Website-aware personalization
│   │   └── ab_testing.py         # A/B testing support
│   │
│   ├── 🔌 integrations/          # Third-party integrations
│   │   ├── webhooks.py           # Webhook server
│   │   ├── api.py                # REST API
│   │   └── sync.py               # Third-party sync
│   │
│   ├── 🛒 marketplace/           # Lead pack marketplace
│   │   ├── store.py              # Marketplace logic
│   │   ├── packs.py              # Lead generation packs
│   │   └── community.py          # Community contributions
│   │
│   └── 🛡️ compliance/            # GDPR & compliance
│       ├── restrictions.py       # Geo-aware scraping restrictions
│       ├── opt_out.py            # Opt-out flows
│       └── retention.py          # Data retention rules
│
├── 🤖 ai/                        # AI/ML components
│   ├── models/                   # Pretrained models
│   │   ├── __init__.py
│   │   ├── tinybert/             # TinyBERT models
│   │   └── scikit/               # Scikit-learn models
│   ├── nlp.py                    # NLP processing engine
│   ├── scoring.py                # Lead scoring system
│   └── classification.py         # Text classification
│
├── 🖥️ interface/                 # User interfaces
│   ├── gui/                      # PyQt5 GUI
│   │   ├── main_window.py        # Main application window
│   │   ├── dashboard.py          # Dashboard view
│   │   ├── components/           # Reusable GUI components
│   │   │   ├── glass_frame.py    # Custom glass frame widget
│   │   │   ├── sidebar.py        # Sidebar widget
│   │   │   ├── stat_card.py      # Statistics card widget
│   │   │   ├── activity_item.py  # Activity list item
│   │   │   ├── campaign_card.py  # Campaign card widget
│   │   │   └── welcome_banner.py # Welcome banner widget
│   │   ├── styles/               # GUI stylesheets
│   │   │   └── glassmorphism.qss # Main stylesheet
│   │   └── resources/            # GUI-specific resources
│   │       └── icons/            # Icons for GUI
│   │
│   └── cli/                      # Command-line interface
│       ├── main.py               # CLI entry point
│       └── commands.py           # CLI commands
├── scripts/                 
│   └── download_models.py
│ 
├── 📦 resources/                 # Static resources
│   ├── templates/                # Email templates
│   ├── icons/                    # UI icons
│   ├── adapters/                 # Pre-built adapters
│   └── styles/                   # GUI stylesheets
│
└── 🚀 deploy/                    # Packaging & deployment
    ├── windows/                  # Windows packaging
    ├── macos/                    # macOS packaging
    └── linux/                    # Linux packaging


    Usage:-
GUI Mode
Launch the application with the graphical interface:
python home/main.py

CLI Mode
Run specific operations from the command line:
python interface/cli/main.py generate-leads --source linkedin --keywords "CTO SaaS"
python interface/cli/main.py send-campaign --campaign-id 12345
python interface/cli/main.py monitor-conversations --platforms reddit,discord

Daemon Mode
Run as a background service:
python home/main.py --daemon

Key Workflows
1. Lead Generation
Open the Lead Generation module
Select sources (LinkedIn, Google, etc.)
Configure search parameters
Start scraping process
Review and export generated leads
2. Lead Enrichment
Import leads or use generated ones
Select enrichment options (social profiles, company data, etc.)
Run enrichment process
Review enhanced lead profiles
3. Email Campaigns
Create a new campaign in the Email System module
Select leads or use persona-based targeting
Choose or create email templates
Configure sending schedule
Launch campaign and monitor results
4. Conversation Mining
Configure platforms to monitor (Reddit, Discord, etc.)
Set up keywords and intent detection rules
Start monitoring
Review high-intent conversations and extracted leads
Licensing & Pricing
License Tiers
Trial: Free for 300 leads, limited features
Standard: $1.8 per 180 leads generated, pay-as-you-go
Admin: Custom pricing with full feature access and team management
License Management
Offline-ready license validation
Role-based feature access
Optional billing integration with Stripe or Paddle
Compliance & Ethics
DRN.today is built with privacy and ethical considerations at its core:

Data Protection
Local Processing: All AI processing occurs locally using TinyBERT, Scikit-learn, and other open-source models
No External APIs: Zero dependency on external AI services for core features
Encrypted Storage: All local data is encrypted with industry-standard algorithms
Compliance Features
GDPR & CCPA Support: Built-in compliance with major data protection regulations
Geo-aware Restrictions: Respects regional scraping laws and restrictions
Opt-out Management: Automated handling of opt-out requests
Data Retention Policies: Configurable data retention and automatic deletion
Ethical Guidelines
Source Blacklisting: Ability to exclude specific domains and sources
Rate Limiting: Respectful scraping with configurable delays
User Agent Transparency: Clear identification in HTTP headers
No Personal Data Misuse: Strict policies against using personal data beyond intended purposes
Contributing
We welcome contributions to DRN.today! Please follow these guidelines:

Development Setup
Fork the repository
Create a feature branch: git checkout -b feature-name
Make your changes and follow the existing code style
Test your changes thoroughly
Submit a pull request with a detailed description
Code Style
Follow PEP 8 guidelines
Use type hints where appropriate
Write docstrings for all public methods and functions
Keep modules focused and cohesive
Testing
Write unit tests for new features
Ensure all tests pass before submitting
Test on all supported platforms (Windows, macOS, Linux)
Documentation
Update documentation for any new features
Add comments to complex code sections
Include examples in docstrings
Support
Documentation
Comprehensive documentation is available at docs.drn.today

Community
Join our community forums at community.drn.today to:

Ask questions
Share adapter configurations
Discuss best practices
Request features
Bug Reports
Report bugs and issues through our GitHub issue tracker. Please include:

Operating system and version
DRN.today version
Steps to reproduce the issue
Expected behavior vs. actual behavior
Screenshots if applicable
Enterprise Support
For enterprise customers, we offer:

Priority email support
Dedicated account manager
Custom adapter development
On-site training
SLA guarantees
Contact enterprise@drn.today for more information.

© 2025 DRN.today. All rights reserved.
