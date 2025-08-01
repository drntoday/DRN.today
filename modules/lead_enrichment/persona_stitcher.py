import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal

from ai.nlp import NLPProcessor
from ai.scoring import LeadScorer
from engine.storage import SecureStorage
from engine.license import LicenseManager

logger = logging.getLogger(__name__)

class PersonaStitcher(QObject):
    """Cross-platform identity resolution and lead enrichment"""
    
    # Signals for UI updates
    enrichment_progress = pyqtSignal(int)
    enrichment_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.nlp_processor = NLPProcessor()
        self.lead_scorer = LeadScorer()
        self.storage = SecureStorage()
        self.license_manager = LicenseManager()
        
        # Identity resolution thresholds
        self.name_similarity_threshold = 0.85
        self.domain_similarity_threshold = 0.9
        self.social_similarity_threshold = 0.7
        
    def enrich(self, input_file: str, enrichment_types: List[str]) -> List[Dict]:
        """Enrich leads with additional data"""
        try:
            # Load leads from input file
            leads = self._load_leads(input_file)
            
            # Process each enrichment type
            for enrichment_type in enrichment_types:
                logger.info(f"Starting enrichment: {enrichment_type}")
                
                if enrichment_type == 'social':
                    leads = self._enrich_social_profiles(leads)
                elif enrichment_type == 'company':
                    leads = self._enrich_company_data(leads)
                elif enrichment_type == 'funding':
                    leads = self._enrich_funding_data(leads)
                elif enrichment_type == 'technologies':
                    leads = self._enrich_technologies(leads)
                elif enrichment_type == 'contacts':
                    leads = self._enrich_contacts(leads)
                
                # Update progress
                progress = (enrichment_types.index(enrichment_type) + 1) / len(enrichment_types) * 100
                self.enrichment_progress.emit(int(progress))
            
            # Save enriched leads
            self._save_enriched_leads(leads, input_file)
            
            # Emit completion signal
            self.enrichment_complete.emit(leads)
            
            return leads
            
        except Exception as e:
            logger.error(f"Error during lead enrichment: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Enrichment failed: {str(e)}")
            raise
    
    def resolve_identity(self, lead_data: Dict) -> Dict:
        """Resolve identity across multiple platforms"""
        try:
            resolved_lead = lead_data.copy()
            
            # Extract key identifiers
            name = lead_data.get('name', '').lower()
            domain = lead_data.get('domain', '').lower()
            email = lead_data.get('email', '').lower()
            
            # Find matching leads in storage
            existing_leads = self.storage.find_leads(
                filters={'name': name, 'domain': domain}
            )
            
            if existing_leads:
                # Merge with existing lead
                resolved_lead = self._merge_leads(resolved_lead, existing_leads[0])
            
            # Perform social profile resolution
            resolved_lead = self._resolve_social_profiles(resolved_lead)
            
            # Add AI-inferred insights
            resolved_lead = self._add_ai_insights(resolved_lead)
            
            return resolved_lead
            
        except Exception as e:
            logger.error(f"Error resolving identity: {str(e)}", exc_info=True)
            raise
    
    def _load_leads(self, input_file: str) -> List[Dict]:
        """Load leads from input file"""
        file_path = Path(input_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _save_enriched_leads(self, leads: List[Dict], original_file: str):
        """Save enriched leads to storage"""
        try:
            # Create enriched filename
            original_path = Path(original_file)
            enriched_path = original_path.parent / f"{original_path.stem}_enriched{original_path.suffix}"
            
            # Save based on file type
            if original_path.suffix.lower() == '.json':
                with open(enriched_path, 'w', encoding='utf-8') as f:
                    json.dump(leads, f, indent=2, ensure_ascii=False)
            elif original_path.suffix.lower() == '.csv':
                df = pd.DataFrame(leads)
                df.to_csv(enriched_path, index=False)
            
            # Also save to database
            self.storage.save_leads(leads)
            
            logger.info(f"Saved enriched leads to {enriched_path}")
            
        except Exception as e:
            logger.error(f"Error saving enriched leads: {str(e)}", exc_info=True)
            raise
    
    def _enrich_social_profiles(self, leads: List[Dict]) -> List[Dict]:
        """Enrich leads with social profile data"""
        enriched_leads = []
        
        for lead in leads:
            try:
                # Find social profiles
                social_profiles = self._find_social_profiles(lead)
                
                # Add to lead data
                lead['social_profiles'] = social_profiles
                
                # Calculate social completeness score
                lead['social_completeness'] = self._calculate_social_completeness(social_profiles)
                
                enriched_leads.append(lead)
                
            except Exception as e:
                logger.warning(f"Error enriching social profiles for lead {lead.get('id', 'unknown')}: {str(e)}")
                enriched_leads.append(lead)  # Keep original lead
        
        return enriched_leads
    
    def _enrich_company_data(self, leads: List[Dict]) -> List[Dict]:
        """Enrich leads with company information"""
        enriched_leads = []
        
        for lead in leads:
            try:
                domain = lead.get('domain')
                if not domain:
                    enriched_leads.append(lead)
                    continue
                
                # Get company data
                company_data = self._get_company_data(domain)
                
                # Add to lead data
                lead['company'] = company_data
                
                # Calculate company fit score
                lead['company_fit_score'] = self._calculate_company_fit(company_data)
                
                enriched_leads.append(lead)
                
            except Exception as e:
                logger.warning(f"Error enriching company data for lead {lead.get('id', 'unknown')}: {str(e)}")
                enriched_leads.append(lead)
        
        return enriched_leads
    
    def _enrich_funding_data(self, leads: List[Dict]) -> List[Dict]:
        """Enrich leads with funding information"""
        enriched_leads = []
        
        for lead in leads:
            try:
                domain = lead.get('domain')
                if not domain:
                    enriched_leads.append(lead)
                    continue
                
                # Get funding data
                funding_data = self._get_funding_data(domain)
                
                # Add to lead data
                lead['funding'] = funding_data
                
                # Calculate funding attractiveness score
                lead['funding_attractiveness'] = self._calculate_funding_attractiveness(funding_data)
                
                enriched_leads.append(lead)
                
            except Exception as e:
                logger.warning(f"Error enriching funding data for lead {lead.get('id', 'unknown')}: {str(e)}")
                enriched_leads.append(lead)
        
        return enriched_leads
    
    def _enrich_technologies(self, leads: List[Dict]) -> List[Dict]:
        """Enrich leads with technology stack information"""
        enriched_leads = []
        
        for lead in leads:
            try:
                domain = lead.get('domain')
                if not domain:
                    enriched_leads.append(lead)
                    continue
                
                # Get technology data
                tech_data = self._get_technology_data(domain)
                
                # Add to lead data
                lead['technologies'] = tech_data
                
                # Calculate tech compatibility score
                lead['tech_compatibility'] = self._calculate_tech_compatibility(tech_data)
                
                enriched_leads.append(lead)
                
            except Exception as e:
                logger.warning(f"Error enriching technology data for lead {lead.get('id', 'unknown')}: {str(e)}")
                enriched_leads.append(lead)
        
        return enriched_leads
    
    def _enrich_contacts(self, leads: List[Dict]) -> List[Dict]:
        """Enrich leads with additional contact information"""
        enriched_leads = []
        
        for lead in leads:
            try:
                # Get additional contacts
                contacts = self._find_additional_contacts(lead)
                
                # Add to lead data
                lead['additional_contacts'] = contacts
                
                # Calculate contact completeness score
                lead['contact_completeness'] = self._calculate_contact_completeness(contacts)
                
                enriched_leads.append(lead)
                
            except Exception as e:
                logger.warning(f"Error enriching contacts for lead {lead.get('id', 'unknown')}: {str(e)}")
                enriched_leads.append(lead)
        
        return enriched_leads
    
    def _find_social_profiles(self, lead: Dict) -> Dict:
        """Find social profiles for a lead"""
        social_profiles = {}
        
        # Implementation for finding social profiles
        # This would typically involve:
        # 1. Searching for profiles by name and company
        # 2. Verifying profile authenticity
        # 3. Extracting profile data
        
        # Placeholder implementation
        social_profiles = {
            'linkedin': None,
            'twitter': None,
            'github': None,
            'facebook': None
        }
        
        return social_profiles
    
    def _get_company_data(self, domain: str) -> Dict:
        """Get company information for a domain"""
        # Implementation for getting company data
        # This would typically involve:
        # 1. Looking up company information
        # 2. Extracting company details
        # 3. Verifying data accuracy
        
        # Placeholder implementation
        return {
            'name': '',
            'industry': '',
            'size': '',
            'revenue': '',
            'description': '',
            'founded': '',
            'location': '',
            'website': domain
        }
    
    def _get_funding_data(self, domain: str) -> Dict:
        """Get funding information for a company"""
        # Implementation for getting funding data
        # This would typically involve:
        # 1. Searching funding databases
        # 2. Extracting funding rounds
        # 3. Calculating total funding
        
        # Placeholder implementation
        return {
            'total_funding': 0,
            'last_round': '',
            'last_round_date': '',
            'investors': [],
            'funding_rounds': []
        }
    
    def _get_technology_data(self, domain: str) -> Dict:
        """Get technology stack information for a domain"""
        # Implementation for getting technology data
        # This would typically involve:
        # 1. Analyzing website technologies
        # 2. Identifying tech stack components
        # 3. Categorizing technologies
        
        # Placeholder implementation
        return {
            'frontend': [],
            'backend': [],
            'analytics': [],
            'hosting': [],
            'cms': [],
            'ecommerce': [],
            'advertising': []
        }
    
    def _find_additional_contacts(self, lead: Dict) -> List[Dict]:
        """Find additional contacts for a company"""
        # Implementation for finding additional contacts
        # This would typically involve:
        # 1. Searching company website
        # 2. Extracting contact information
        # 3. Verifying contact details
        
        # Placeholder implementation
        return []
    
    def _merge_leads(self, lead1: Dict, lead2: Dict) -> Dict:
        """Merge two lead records"""
        merged = lead1.copy()
        
        # Merge fields intelligently
        for key, value in lead2.items():
            if key not in merged or not merged[key]:
                merged[key] = value
            elif key == 'social_profiles':
                # Merge social profiles
                merged[key] = self._merge_social_profiles(merged[key], value)
            elif key == 'technologies':
                # Merge technology lists
                merged[key] = self._merge_technology_lists(merged[key], value)
        
        return merged
    
    def _merge_social_profiles(self, profiles1: Dict, profiles2: Dict) -> Dict:
        """Merge social profile dictionaries"""
        merged = profiles1.copy()
        
        for platform, url in profiles2.items():
            if not merged.get(platform):
                merged[platform] = url
        
        return merged
    
    def _merge_technology_lists(self, tech1: Dict, tech2: Dict) -> Dict:
        """Merge technology category dictionaries"""
        merged = tech1.copy()
        
        for category, technologies in tech2.items():
            if category not in merged:
                merged[category] = technologies
            else:
                # Merge technology lists, removing duplicates
                merged[category] = list(set(merged[category] + technologies))
        
        return merged
    
    def _resolve_social_profiles(self, lead: Dict) -> Dict:
        """Resolve social profiles across platforms"""
        # Implementation for social profile resolution
        # This would typically involve:
        # 1. Cross-referencing profiles
        # 2. Verifying profile ownership
        # 3. Consolidating profile data
        
        return lead
    
    def _add_ai_insights(self, lead: Dict) -> Dict:
        """Add AI-inferred insights to lead"""
        try:
            # Calculate lead score
            lead_score = self.lead_scorer.score_lead(lead)
            lead['ai_score'] = lead_score
            
            # Add persona insights
            persona_insights = self.nlp_processor.analyze_persona(lead)
            lead['persona_insights'] = persona_insights
            
            # Add predicted attributes
            predicted_attrs = self.nlp_processor.predict_attributes(lead)
            lead.update(predicted_attrs)
            
            return lead
            
        except Exception as e:
            logger.error(f"Error adding AI insights: {str(e)}", exc_info=True)
            return lead
    
    def _calculate_social_completeness(self, social_profiles: Dict) -> float:
        """Calculate social profile completeness score"""
        total_platforms = len(social_profiles)
        if total_platforms == 0:
            return 0.0
        
        completed_platforms = sum(1 for url in social_profiles.values() if url)
        return completed_platforms / total_platforms
    
    def _calculate_company_fit(self, company_data: Dict) -> float:
        """Calculate company fit score"""
        # Implementation for company fit calculation
        # This would typically involve:
        # 1. Evaluating company size
        # 2. Assessing industry relevance
        # 3. Considering funding stage
        
        return 0.5  # Placeholder
    
    def _calculate_funding_attractiveness(self, funding_data: Dict) -> float:
        """Calculate funding attractiveness score"""
        # Implementation for funding attractiveness calculation
        # This would typically involve:
        # 1. Evaluating total funding
        # 2. Considering funding stage
        # 3. Assessing investor quality
        
        return 0.5  # Placeholder
    
    def _calculate_tech_compatibility(self, tech_data: Dict) -> float:
        """Calculate technology compatibility score"""
        # Implementation for tech compatibility calculation
        # This would typically involve:
        # 1. Matching tech stack
        # 2. Evaluating tech preferences
        # 3. Considering integration potential
        
        return 0.5  # Placeholder
    
    def _calculate_contact_completeness(self, contacts: List[Dict]) -> float:
        """Calculate contact completeness score"""
        if not contacts:
            return 0.0
        
        # Calculate based on contact information completeness
        total_completeness = 0
        for contact in contacts:
            completeness = 0
            if contact.get('name'):
                completeness += 0.25
            if contact.get('email'):
                completeness += 0.25
            if contact.get('phone'):
                completeness += 0.25
            if contact.get('title'):
                completeness += 0.25
            total_completeness += completeness
        
        return total_completeness / len(contacts)
