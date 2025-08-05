#!/usr/bin/env python3
"""
Domain Classifier for NanoBrain Framework
Classifies scientific domain from natural language requests using configurable methods.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional, List, Set
import re

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.universal_models import DomainClassification, DomainType
from pydantic import Field

# Domain classifier logger
logger = logging.getLogger(__name__)


class DomainClassifierConfig(ConfigBase):
    """Configuration for domain classifier"""
    
    # Classification method
    method: str = Field(
        default="keyword_based",
        description="Classification method to use"
    )
    
    # Confidence thresholds
    confidence_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        },
        description="Confidence thresholds for classification"
    )
    
    # Domain-specific patterns (component expects this field name)
    domain_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            'bioinformatics': [
                'protein', 'sequence', 'genome', 'virus', 'viral',
                'biological', 'molecular', 'genetic'
            ],
            'analysis': [
                'analyze', 'analysis', 'statistical', 'computation', 'calculate'
            ],
            'data_processing': [
                'process', 'transform', 'convert', 'format', 'parse'
            ]
        },
        description="Domain-specific pattern mappings"
    )
    
    # Method-specific configurations
    keyword_based_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'case_sensitive': False,
            'minimum_keyword_matches': 1,
            'boost_primary_domain': True
        },
        description="Keyword-based classification configuration"
    )
    
    # Semantic configuration
    semantic_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'domain_embeddings': {},
            'enable_preprocessing': True
        },
        description="Semantic classification configuration"
    )
    
    # Classification behavior
    classification_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'allow_multiple_domains': True,
            'minimum_keyword_matches': 1,
            'boost_primary_domain': True
        },
        description="General classification behavior settings"
    )
    
    # Performance settings
    performance_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_caching': True,
            'cache_ttl_seconds': 1800,
            'case_sensitive': False
        },
        description="Performance optimization settings"
    )


class DomainClassifier(FromConfigBase):
    """Configurable domain classification for request routing"""
    
    def __init__(self):
        """Initialize domain classifier - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return DomainClassifierConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize classifier from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[DomainClassifierConfig] = None
        self.compiled_patterns: Dict[str, Dict[str, List[re.Pattern]]] = {}
        
        logger.info("🔬 Initializing Domain Classifier")
        self.config = config
        
        # Compile patterns for efficiency
        self.compile_domain_patterns()
        
        logger.info(f"✅ Domain Classifier initialized with {self.config.method} method")
    
    def compile_domain_patterns(self) -> None:
        """Compile regex patterns for efficient domain matching"""
        try:
            for domain_name, pattern_config in self.config.domain_patterns.items():
                domain_patterns = {}
                
                for pattern_type in ['keywords', 'technical_terms', 'organisms', 'methods', 'tools']:
                    if pattern_type in pattern_config:
                        compiled_list = []
                        for term in pattern_config[pattern_type]:
                            # Create word boundary pattern for precise matching
                            pattern = r'\b' + re.escape(term) + r'\b'
                            compiled_list.append(re.compile(pattern, re.IGNORECASE))
                        domain_patterns[pattern_type] = compiled_list
                
                self.compiled_patterns[domain_name] = domain_patterns
            
            logger.debug(f"✅ Compiled patterns for {len(self.compiled_patterns)} domains")
            
        except Exception as e:
            logger.error(f"❌ Pattern compilation failed: {e}")
            raise
    
    async def classify_domain(self, query: str) -> DomainClassification:
        """
        Classify domain using configured method.
        
        Args:
            query: Natural language query to classify
            
        Returns:
            DomainClassification with detected domain and confidence
        """
        try:
            if self.config.method == 'semantic_analysis':
                return self.classify_domain_semantic(query)
            elif self.config.method == 'keyword_based':
                return self.classify_domain_keyword_based(query)
            elif self.config.method == 'hybrid':
                return self.classify_domain_hybrid(query)
            else:
                logger.warning(f"⚠️ Unknown classification method: {self.config.method}")
                return self.classify_domain_semantic(query)
                
        except Exception as e:
            logger.error(f"❌ Domain classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def classify_domain_semantic(self, query: str) -> DomainClassification:
        """Classify domain using semantic analysis approach"""
        try:
            domain_scores = {}
            all_indicators = []
            
            # Analyze each domain
            for domain_name, pattern_config in self.config.domain_patterns.items():
                score = 0.0
                domain_indicators = []
                
                if domain_name in self.compiled_patterns:
                    # Use compiled patterns for matching
                    for pattern_type, patterns in self.compiled_patterns[domain_name].items():
                        for pattern in patterns:
                            matches = pattern.findall(query)
                            if matches:
                                # Weight different pattern types differently
                                type_weight = self.get_pattern_type_weight(pattern_type)
                                match_score = len(matches) * type_weight
                                score += match_score
                                domain_indicators.extend(matches)
                
                # Apply domain weight
                weighted_score = score * pattern_config.get('weight', 1.0)
                
                # Context weighting if enabled
                if self.config.semantic_config.get('use_context_weighting', True):
                    context_multiplier = self.calculate_context_weight(query, domain_indicators)
                    weighted_score *= context_multiplier
                
                # Term frequency weighting if enabled
                if self.config.semantic_config.get('term_frequency_weighting', True):
                    tf_multiplier = self.calculate_term_frequency_weight(query, domain_indicators)
                    weighted_score *= tf_multiplier
                
                if weighted_score > 0:
                    domain_scores[domain_name] = weighted_score
                    all_indicators.extend(domain_indicators)
            
            # Handle multi-domain detection
            if self.config.semantic_config.get('multi_domain_detection', True):
                result = self.handle_multi_domain_result(domain_scores, all_indicators)
            else:
                result = self.handle_single_domain_result(domain_scores, all_indicators)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Semantic classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def classify_domain_keyword_based(self, query: str) -> DomainClassification:
        """Classify domain using simple keyword-based approach"""
        try:
            domain_scores = {}
            found_indicators = []
            
            query_lower = query.lower()
            
            # Simple keyword matching for each domain
            for domain_name, pattern_config in self.config.domain_patterns.items():
                score = 0.0
                domain_indicators = []
                
                # Check all keyword types
                for pattern_type in ['keywords', 'technical_terms', 'organisms', 'methods', 'tools']:
                    if pattern_type in pattern_config:
                        for term in pattern_config[pattern_type]:
                            if term.lower() in query_lower:
                                score += 1.0
                                domain_indicators.append(term)
                
                if score > 0:
                    # Apply domain weight
                    weighted_score = score * pattern_config.get('weight', 1.0)
                    domain_scores[domain_name] = weighted_score
                    found_indicators.extend(domain_indicators)
            
            return self.handle_single_domain_result(domain_scores, found_indicators)
            
        except Exception as e:
            logger.error(f"❌ Keyword-based classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def classify_domain_hybrid(self, query: str) -> DomainClassification:
        """Classify domain using hybrid approach combining semantic and keyword methods"""
        try:
            # Get classifications from both methods
            semantic_result = self.classify_domain_semantic(query)
            keyword_result = self.classify_domain_keyword_based(query)
            
            # Combine results with weighted average
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            # If both methods agree, increase confidence
            if semantic_result.domain_type == keyword_result.domain_type:
                combined_confidence = min(1.0,
                    (semantic_result.confidence * semantic_weight +
                     keyword_result.confidence * keyword_weight) * 1.2  # Boost for agreement
                )
                domain_type = semantic_result.domain_type
                combined_indicators = list(set(semantic_result.indicators + keyword_result.indicators))
            else:
                # Use the result with higher confidence
                if semantic_result.confidence >= keyword_result.confidence:
                    domain_type = semantic_result.domain_type
                    combined_confidence = semantic_result.confidence * 0.9  # Slight penalty for disagreement
                    combined_indicators = semantic_result.indicators
                else:
                    domain_type = keyword_result.domain_type
                    combined_confidence = keyword_result.confidence * 0.9
                    combined_indicators = keyword_result.indicators
            
            return DomainClassification(
                domain_type=domain_type,
                confidence=combined_confidence,
                indicators=combined_indicators,
                classification_method='hybrid',
                metadata={
                    'semantic_result': semantic_result.dict(),
                    'keyword_result': keyword_result.dict(),
                    'methods_agreed': semantic_result.domain_type == keyword_result.domain_type
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Hybrid classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def get_pattern_type_weight(self, pattern_type: str) -> float:
        """Get weight for different pattern types"""
        weights = {
            'technical_terms': 1.5,  # Technical terms are highly indicative
            'organisms': 1.3,        # Specific organisms are strong indicators
            'methods': 1.2,          # Scientific methods are good indicators
            'tools': 1.1,            # Tools are moderately indicative
            'keywords': 1.0          # General keywords have base weight
        }
        return weights.get(pattern_type, 1.0)
    
    def calculate_context_weight(self, query: str, indicators: List[str]) -> float:
        """Calculate context weight based on indicator proximity and co-occurrence"""
        if not indicators or len(indicators) < 2:
            return 1.0
        
        # Simple proximity scoring: if multiple indicators appear close together, boost score
        query_lower = query.lower()
        positions = []
        
        for indicator in indicators:
            pos = query_lower.find(indicator.lower())
            if pos != -1:
                positions.append(pos)
        
        if len(positions) >= 2:
            # Calculate average distance between indicators
            avg_distance = sum(abs(positions[i] - positions[i-1]) 
                             for i in range(1, len(positions))) / (len(positions) - 1)
            
            # Closer indicators get higher weight
            if avg_distance < 50:  # Characters
                return 1.3
            elif avg_distance < 100:
                return 1.1
        
        return 1.0
    
    def calculate_term_frequency_weight(self, query: str, indicators: List[str]) -> float:
        """Calculate weight based on term frequency"""
        if not indicators:
            return 1.0
        
        query_lower = query.lower()
        total_occurrences = 0
        
        for indicator in indicators:
            total_occurrences += query_lower.count(indicator.lower())
        
        # More frequent terms get higher weight
        if total_occurrences >= 3:
            return 1.2
        elif total_occurrences >= 2:
            return 1.1
        
        return 1.0
    
    def handle_single_domain_result(self, domain_scores: Dict[str, float], 
                                  indicators: List[str]) -> DomainClassification:
        """Handle single domain classification result"""
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            raw_confidence = domain_scores[best_domain]
            
            # Normalize confidence score
            confidence = min(1.0, raw_confidence / 5.0)  # Scale for realistic confidence
            
            # Map domain name to enum
            domain_type = self.map_domain_name_to_enum(best_domain)
            
            return DomainClassification(
                domain_type=domain_type,
                confidence=confidence,
                indicators=list(set(indicators)),
                classification_method=self.config.method,
                metadata={
                    'all_scores': domain_scores,
                    'best_score': raw_confidence
                }
            )
        else:
            # No domain detected
            return DomainClassification(
                domain_type=DomainType.UNKNOWN,
                confidence=0.2,
                indicators=[],
                classification_method=self.config.method,
                metadata={'no_patterns_matched': True}
            )
    
    def handle_multi_domain_result(self, domain_scores: Dict[str, float], 
                                 indicators: List[str]) -> DomainClassification:
        """Handle multi-domain classification result"""
        if not domain_scores:
            return self.handle_single_domain_result(domain_scores, indicators)
        
        # Check if multiple domains have significant scores
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_domains) >= 2:
            best_score = sorted_domains[0][1]
            second_score = sorted_domains[1][1]
            
            # If scores are close, it might be a multi-domain query
            if second_score >= best_score * 0.7:
                # This is a potential multi-domain query
                primary_domain = sorted_domains[0][0]
                confidence = min(1.0, best_score / 5.0) * 0.9  # Slight penalty for ambiguity
                
                domain_type = self.map_domain_name_to_enum(primary_domain)
                
                return DomainClassification(
                    domain_type=domain_type,
                    confidence=confidence,
                    indicators=list(set(indicators)),
                    classification_method=self.config.method,
                    metadata={
                        'multi_domain_detected': True,
                        'secondary_domains': [d[0] for d in sorted_domains[1:3]],
                        'all_scores': domain_scores
                    }
                )
        
        # Fall back to single domain handling
        return self.handle_single_domain_result(domain_scores, indicators)
    
    def map_domain_name_to_enum(self, domain_name: str) -> DomainType:
        """Map domain name to DomainType enum"""
        mapping = {
            'virology': DomainType.VIROLOGY,
            'protein_analysis': DomainType.PROTEIN_ANALYSIS,
            'bioinformatics': DomainType.BIOINFORMATICS,
            'genomics': DomainType.GENOMICS,
            'general_science': DomainType.GENERAL_SCIENCE,
            'conversation': DomainType.CONVERSATION
        }
        
        return mapping.get(domain_name, DomainType.UNKNOWN)
    
    def create_fallback_classification(self, query: str) -> DomainClassification:
        """Create fallback classification when main classification fails"""
        return DomainClassification(
            domain_type=DomainType.GENERAL_SCIENCE,
            confidence=0.2,
            indicators=[],
            classification_method='fallback',
            metadata={'error_occurred': True}
        )
    
    def get_supported_domains(self) -> List[DomainType]:
        """Get list of supported domain types"""
        return [
            DomainType.VIROLOGY,
            DomainType.PROTEIN_ANALYSIS,
            DomainType.BIOINFORMATICS,
            DomainType.GENOMICS,
            DomainType.GENERAL_SCIENCE,
            DomainType.CONVERSATION
        ]
    
    def update_domain_patterns(self, new_patterns: Dict[str, Dict[str, Any]]) -> None:
        """Update domain patterns dynamically"""
        try:
            self.config.domain_patterns.update(new_patterns)
            self.compile_domain_patterns()
            logger.info(f"✅ Updated domain patterns for {len(new_patterns)} domains")
        except Exception as e:
            logger.error(f"❌ Failed to update domain patterns: {e}")
            raise 