"""
Virus Name Resolution System for BV-BRC Tool

Resolves user-provided virus names to exact taxon IDs from BVBRC_genome_alphavirus.csv
with fuzzy matching and synonym support.

Based on BV-BRC CLI documentation and data acquisition patterns.
"""

import pandas as pd
import re
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from fuzzywuzzy import fuzz, process

from nanobrain.core.logging_system import get_logger


@dataclass
class TaxonResolution:
    """Result of virus name resolution to taxon ID"""
    taxon_id: str
    matched_name: str
    confidence: int
    match_type: str  # "exact", "fuzzy", "synonym"
    synonyms: List[str] = field(default_factory=list)
    all_genome_names: List[str] = field(default_factory=list)


@dataclass 
class TaxonInfo:
    """Information about a specific taxon"""
    taxon_id: str
    primary_name: str
    synonyms: List[str]
    genome_count: int
    representative_genomes: List[str]


class VirusNameResolver:
    """
    Resolve user virus names to taxon IDs with fuzzy matching.
    
    This system addresses the challenge of mapping user-provided virus names
    to the exact taxon IDs needed for BV-BRC CLI commands like:
    p3-all-genomes --eq taxon_id,<taxon_id>
    """
    
    def __init__(self, csv_path: str = "data/alphavirus_analysis/BVBRC_genome_alphavirus.csv"):
        self.csv_path = Path(csv_path)
        self.taxon_cache: Dict[str, TaxonInfo] = {}
        self.name_index: Dict[str, str] = {}  # virus_name -> taxon_id
        self.synonyms: Dict[str, List[str]] = {}  # taxon_id -> [synonyms]
        self.initialized = False
        self.logger = get_logger("virus_name_resolver")
        
    async def initialize_virus_index(self) -> None:
        """
        Load and index all virus names and synonyms from CSV.
        
        Extracts taxon IDs from genome IDs (e.g., "11020.100" -> "11020")
        and builds comprehensive name index for fuzzy matching.
        """
        if self.initialized:
            return
            
        self.logger.info("🔄 Initializing virus name index from alphavirus CSV")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Alphavirus CSV not found at {self.csv_path}")
        
        try:
            # Load CSV data
            df = pd.read_csv(self.csv_path)
            self.logger.info(f"📊 Loaded {len(df)} genome entries from CSV")
            
            # Extract taxon_id from Genome ID (e.g., "11020.100" -> "11020")
            df['taxon_id'] = df['Genome ID'].str.split('.').str[0]
            
            # Group by taxon_id to get all virus names for each taxon
            taxon_count = 0
            for taxon_id, group in df.groupby('taxon_id'):
                virus_names = group['Genome Name'].unique()
                primary_name = virus_names[0]  # Use first as primary
                
                # Store primary name mapping
                self.name_index[primary_name.lower()] = taxon_id
                self.synonyms[taxon_id] = list(virus_names)
                
                # Generate and store name variations
                all_variations = set()
                for name in virus_names:
                    variations = self._generate_name_variations(name)
                    all_variations.update(variations)
                    
                # Add all variations to index (if not already present)
                for variation in all_variations:
                    variation_lower = variation.lower()
                    if variation_lower not in self.name_index:
                        self.name_index[variation_lower] = taxon_id
                
                # Store taxon info
                self.taxon_cache[taxon_id] = TaxonInfo(
                    taxon_id=taxon_id,
                    primary_name=primary_name,
                    synonyms=list(virus_names),
                    genome_count=len(group),
                    representative_genomes=group['Genome ID'].head(3).tolist()
                )
                
                taxon_count += 1
            
            self.initialized = True
            self.logger.info(f"✅ Virus index initialized: {taxon_count} taxa, {len(self.name_index)} name variations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize virus index: {e}")
            raise
    
    async def resolve_virus_name(self, user_input: str, confidence_threshold: int = 80) -> Optional[TaxonResolution]:
        """
        Resolve user virus name to taxon ID with confidence score.
        
        Args:
            user_input: User-provided virus name (e.g., "CHIKV", "Chikungunya virus")
            confidence_threshold: Minimum confidence score for fuzzy matches
            
        Returns:
            TaxonResolution object with taxon_id and match details, or None if no match
        """
        if not self.initialized:
            await self.initialize_virus_index()
        
        if not user_input or not user_input.strip():
            return None
            
        user_input_clean = user_input.strip()
        self.logger.info(f"🔍 Resolving virus name: '{user_input_clean}'")
        
        # Try exact match first (case-insensitive)
        exact_match = self.name_index.get(user_input_clean.lower())
        if exact_match:
            taxon_info = self.taxon_cache[exact_match]
            self.logger.info(f"✅ Exact match found: {exact_match} -> {taxon_info.primary_name}")
            
            return TaxonResolution(
                taxon_id=exact_match,
                matched_name=taxon_info.primary_name,
                confidence=100,
                match_type="exact",
                synonyms=taxon_info.synonyms,
                all_genome_names=taxon_info.synonyms
            )
        
        # Try fuzzy matching
        self.logger.debug(f"No exact match, trying fuzzy matching with threshold {confidence_threshold}%")
        
        try:
            best_match, confidence = process.extractOne(
                user_input_clean.lower(), 
                self.name_index.keys(),
                scorer=fuzz.token_sort_ratio
            )
            
            if confidence >= confidence_threshold:
                taxon_id = self.name_index[best_match]
                taxon_info = self.taxon_cache[taxon_id]
                
                self.logger.info(f"✅ Fuzzy match found: '{user_input_clean}' -> '{best_match}' "
                                f"(confidence: {confidence}%, taxon: {taxon_id})")
                
                return TaxonResolution(
                    taxon_id=taxon_id,
                    matched_name=taxon_info.primary_name,
                    confidence=confidence,
                    match_type="fuzzy",
                    synonyms=taxon_info.synonyms,
                    all_genome_names=taxon_info.synonyms
                )
            else:
                self.logger.warning(f"⚠️ Best fuzzy match '{best_match}' has low confidence: {confidence}% "
                                   f"(threshold: {confidence_threshold}%)")
        
        except Exception as e:
            self.logger.error(f"❌ Error during fuzzy matching: {e}")
        
        self.logger.warning(f"❌ No suitable match found for '{user_input_clean}'")
        return None
    
    def _generate_name_variations(self, virus_name: str) -> List[str]:
        """
        Generate common variations of virus names for better matching.
        
        Handles cases like:
        - "Chikungunya virus" -> "Chikungunya", "CHIKV"
        - "Eastern equine encephalitis virus" -> "Eastern equine encephalitis", "EEEV"
        - Strain/isolate removal
        """
        variations = [virus_name]
        
        # Remove "virus" suffix
        if virus_name.lower().endswith(' virus'):
            base_name = virus_name[:-6].strip()
            variations.append(base_name)
        
        # Handle acronyms (e.g., "EEEV" for "Eastern equine encephalitis virus")
        words = virus_name.split()
        if len(words) > 1:
            # Create acronym from first letters of significant words
            significant_words = [word for word in words if word.lower() not in ['virus', 'strain', 'isolate']]
            if len(significant_words) >= 2:
                acronym = ''.join(word[0].upper() for word in significant_words)
                if len(acronym) >= 2:
                    variations.append(acronym)
                    
                # Also try with 'V' suffix for virus
                if not virus_name.lower().endswith(' virus'):
                    variations.append(acronym + 'V')
        
        # Remove strain/isolate information
        # Patterns: "strain X", "isolate Y", "var. Z", "variant W"
        base_name = re.sub(r'\s+(strain|isolate|var\.|variant)\s+.*', '', virus_name, flags=re.IGNORECASE)
        if base_name != virus_name:
            variations.append(base_name.strip())
        
        # Remove parenthetical information
        base_name = re.sub(r'\s*\([^)]*\)', '', virus_name)
        if base_name != virus_name:
            variations.append(base_name.strip())
        
        # Handle common abbreviations
        abbreviation_map = {
            'equine': 'eq',
            'encephalitis': 'enceph',
            'venezuelan': 'ven',
            'eastern': 'east',
            'western': 'west'
        }
        
        for full_word, abbrev in abbreviation_map.items():
            if full_word in virus_name.lower():
                abbreviated = virus_name.lower().replace(full_word, abbrev)
                variations.append(abbreviated)
        
        # Remove duplicates and empty strings
        variations = list(set(v.strip() for v in variations if v.strip()))
        
        return variations
    
    async def get_available_taxa(self) -> List[TaxonInfo]:
        """Get list of all available taxa for reference."""
        if not self.initialized:
            await self.initialize_virus_index()
        
        return list(self.taxon_cache.values())
    
    async def suggest_similar_names(self, user_input: str, max_suggestions: int = 5) -> List[Tuple[str, int]]:
        """
        Suggest similar virus names when no good match is found.
        
        Returns list of (virus_name, confidence_score) tuples.
        """
        if not self.initialized:
            await self.initialize_virus_index()
        
        if not user_input or not user_input.strip():
            return []
        
        try:
            # Get top fuzzy matches regardless of threshold
            matches = process.extract(
                user_input.lower(),
                self.name_index.keys(),
                scorer=fuzz.token_sort_ratio,
                limit=max_suggestions
            )
            
            suggestions = []
            for match_name, confidence in matches:
                taxon_id = self.name_index[match_name]
                taxon_info = self.taxon_cache[taxon_id]
                suggestions.append((taxon_info.primary_name, confidence))
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"❌ Error generating suggestions: {e}")
            return []
    
    def get_taxon_info(self, taxon_id: str) -> Optional[TaxonInfo]:
        """Get detailed information about a specific taxon."""
        return self.taxon_cache.get(taxon_id) 