#!/usr/bin/env python3
"""
Universal Request Analyzer for NanoBrain Framework
Analyzes natural language requests for intent, domain, and workflow routing decisions.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.request_models import ChatRequest
from nanobrain.library.interfaces.web.models.universal_models import (
    RequestAnalysis, IntentClassification, DomainClassification,
    IntentType, DomainType
)
from pydantic import Field

# Analyzer logger
logger = logging.getLogger(__name__)


class RequestAnalyzerConfig(ConfigBase):
    """Configuration for universal request analyzer"""
    
    # Analysis method configuration
    analysis_method: str = Field(
        default='comprehensive',
        description="Analysis method to use"
    )
    
    # Component configurations (Allow Any type for framework object resolution)
    intent_classifier: Any = Field(
        default_factory=lambda: {
            'class': 'nanobrain.library.interfaces.web.analysis.intent_classifier.IntentClassifier',
            'config': 'nanobrain/library/interfaces/web/config/intent_classifier_config.yml'
        },
        description="Intent classifier configuration or resolved object"
    )
    
    domain_classifier: Any = Field(
        default_factory=lambda: {
            'class': 'nanobrain.library.interfaces.web.analysis.domain_classifier.DomainClassifier',
            'config': 'nanobrain/library/interfaces/web/config/domain_classifier_config.yml'
        },
        description="Domain classifier configuration or resolved object"
    )
    
    # Analysis thresholds
    confidence_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'intent_minimum': 0.3,
            'domain_minimum': 0.3,
            'analysis_minimum': 0.5
        },
        description="Confidence thresholds for analysis"
    )
    
    # Entity extraction configuration
    entity_extraction: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enabled': True,
            'extract_virus_names': True,
            'extract_protein_sequences': True,
            'extract_analysis_types': True
        },
        description="Entity extraction settings"
    )
    
    # Performance settings
    performance_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'parallel_analysis': True
        },
        description="Performance optimization settings"
    )


class UniversalRequestAnalyzer(FromConfigBase):
    """
    Configurable natural language request analysis.
    Analyzes requests for intent, domain, and workflow routing decisions.
    """
    
    def __init__(self):
        """Initialize request analyzer - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return RequestAnalyzerConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize analyzer from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[RequestAnalyzerConfig] = None
        self.intent_classifier: Optional[Any] = None
        self.domain_classifier: Optional[Any] = None
        self.analysis_cache: Dict[str, RequestAnalysis] = {}
        
        logger.info("🔍 Initializing Universal Request Analyzer")
        self.config = config
        
        # Load analysis components
        self.load_analysis_components()
        
        # Setup performance optimizations
        self.setup_performance_optimizations()
        
        logger.info("✅ Universal Request Analyzer initialized successfully")
    
    def load_analysis_components(self) -> None:
        """Load and initialize analysis components"""
        try:
            logger.debug("🔧 Loading analysis components")
            
            # Load intent classifier (check if already resolved by framework)
            intent_config = self.config.intent_classifier
            if hasattr(intent_config, 'classify_intent'):
                # ✅ FRAMEWORK COMPLIANCE: Already resolved by nested object resolution
                self.intent_classifier = intent_config
                logger.debug("✅ Intent classifier already resolved by framework")
            else:
                # Load via framework pattern (fallback for dictionary config)
                self.intent_classifier = self.load_component_via_framework('intent_classifier', intent_config)
            
            # Load domain classifier (check if already resolved by framework)
            domain_config = self.config.domain_classifier
            if hasattr(domain_config, 'classify_domain'):
                # ✅ FRAMEWORK COMPLIANCE: Already resolved by nested object resolution
                self.domain_classifier = domain_config
                logger.debug("✅ Domain classifier already resolved by framework")
            else:
                # Load via framework pattern (fallback for dictionary config)
                self.domain_classifier = self.load_component_via_framework('domain_classifier', domain_config)
            
            logger.debug("✅ Analysis components loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load analysis components: {e}")
            raise
    
    def load_component_via_framework(self, component_name: str, component_config: Dict[str, Any]) -> Any:
        """Load a component using NanoBrain framework patterns"""
        try:
            # Extract component class and configuration
            component_class_path = component_config.get('class')
            component_cfg = component_config.get('config', component_config)
            
            if not component_class_path:
                raise ValueError(f"Component '{component_name}' missing required 'class' field")
            
            # Import component class dynamically
            module_path, class_name = component_class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Create component using from_config pattern
            if hasattr(component_class, 'from_config'):
                component = component_class.from_config(component_cfg)
                logger.debug(f"✅ Component '{component_name}' loaded via from_config")
                return component
            else:
                # Fallback to direct instantiation for simple components
                component = component_class(component_cfg)
                logger.debug(f"✅ Component '{component_name}' loaded via direct instantiation")
                return component
                
        except Exception as e:
            logger.error(f"❌ Failed to load component '{component_name}': {e}")
            raise
    
    def setup_performance_optimizations(self) -> None:
        """Setup performance optimizations"""
        if self.config.performance_config.get('enable_caching', True):
            logger.debug("✅ Analysis caching enabled")
        
        if self.config.performance_config.get('parallel_analysis', True):
            logger.debug("✅ Parallel analysis enabled")
    
    async def analyze_request(self, request: ChatRequest) -> RequestAnalysis:
        """
        Analyze natural language request for workflow routing.
        
        Args:
            request: ChatRequest to analyze
            
        Returns:
            Complete RequestAnalysis with intent, domain, and entities
        """
        try:
            analysis_start = datetime.now()
            request_id = request.request_id or str(uuid.uuid4())
            
            logger.debug(f"🔍 Analyzing request: {request_id}")
            
            # Check cache first
            if self.config.performance_config.get('enable_caching', True):
                cached_analysis = self.get_cached_analysis(request.query)
                if cached_analysis:
                    logger.debug(f"✅ Retrieved cached analysis for: {request_id}")
                    return cached_analysis
            
            # Perform parallel analysis if enabled
            if self.config.performance_config.get('parallel_analysis', True):
                intent_task = self.extract_intent_async(request.query)
                domain_task = self.identify_domain_async(request.query)
                entities_task = self.extract_entities_async(request.query)
                
                # Wait for all analyses to complete
                intent_classification, domain_classification, extracted_entities = await asyncio.gather(
                    intent_task, domain_task, entities_task
                )
            else:
                # Sequential analysis
                intent_classification = await self.extract_intent_async(request.query)
                domain_classification = await self.identify_domain_async(request.query)
                extracted_entities = await self.extract_entities_async(request.query)
            
            # Calculate complexity score
            complexity_score = self.calculate_complexity_score(
                request.query, intent_classification, domain_classification, extracted_entities
            )
            
            # Create complete analysis
            analysis = RequestAnalysis(
                request_id=request_id,
                original_query=request.query,
                intent_classification=intent_classification,
                domain_classification=domain_classification,
                extracted_entities=extracted_entities,
                complexity_score=complexity_score,
                analysis_timestamp=datetime.now(),
                analyzer_metadata={
                    'analysis_method': self.config.analysis_method,
                    'processing_time_ms': (datetime.now() - analysis_start).total_seconds() * 1000,
                    'cached': False,
                    'parallel_analysis': self.config.performance_config.get('parallel_analysis', True)
                }
            )
            
            # Cache analysis if enabled
            if self.config.performance_config.get('enable_caching', True):
                self.cache_analysis(request.query, analysis)
            
            logger.debug(f"✅ Request analysis completed: {request_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Request analysis failed: {e}")
            # Return fallback analysis
            return self.create_fallback_analysis(request)
    
    async def extract_intent_async(self, query: str) -> IntentClassification:
        """Extract user intent using configured classification method"""
        try:
            if self.intent_classifier:
                # Use configured intent classifier
                classification = await self.intent_classifier.classify_intent(query)
                
                # Validate confidence threshold
                if classification.confidence < self.config.confidence_thresholds.get('intent_minimum', 0.3):
                    logger.debug(f"⚠️ Intent confidence below threshold: {classification.confidence}")
                    classification.intent_type = IntentType.UNKNOWN
                
                return classification
            else:
                # Fallback intent classification
                return self.fallback_intent_classification(query)
                
        except Exception as e:
            logger.error(f"❌ Intent extraction failed: {e}")
            return self.fallback_intent_classification(query)
    
    async def identify_domain_async(self, query: str) -> DomainClassification:
        """Identify domain using configured classification strategy"""
        try:
            if self.domain_classifier:
                # Use configured domain classifier
                classification = await self.domain_classifier.classify_domain(query)
                
                # Validate confidence threshold
                if classification.confidence < self.config.confidence_thresholds.get('domain_minimum', 0.3):
                    logger.debug(f"⚠️ Domain confidence below threshold: {classification.confidence}")
                    classification.domain_type = DomainType.UNKNOWN
                
                return classification
            else:
                # Fallback domain classification
                return self.fallback_domain_classification(query)
                
        except Exception as e:
            logger.error(f"❌ Domain identification failed: {e}")
            return self.fallback_domain_classification(query)
    
    async def extract_entities_async(self, query: str) -> Dict[str, Any]:
        """Extract entities from query based on configuration"""
        try:
            entities = {}
            
            if not self.config.entity_extraction.get('enabled', True):
                return entities
            
            query_lower = query.lower()
            
            # Extract virus names if enabled
            if self.config.entity_extraction.get('extract_virus_names', True):
                virus_names = self.extract_virus_names(query_lower)
                if virus_names:
                    entities['virus_names'] = virus_names
            
            # Extract protein sequences if enabled
            if self.config.entity_extraction.get('extract_protein_sequences', True):
                protein_sequences = self.extract_protein_sequences(query)
                if protein_sequences:
                    entities['protein_sequences'] = protein_sequences
            
            # Extract analysis types if enabled
            if self.config.entity_extraction.get('extract_analysis_types', True):
                analysis_types = self.extract_analysis_types(query_lower)
                if analysis_types:
                    entities['analysis_types'] = analysis_types
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return {}
    
    def extract_virus_names(self, query_lower: str) -> List[str]:
        """Extract virus names from query"""
        virus_patterns = [
            'eeev', 'eastern equine encephalitis', 'chikungunya', 'chikv',
            'zika', 'dengue', 'yellow fever', 'alphavirus', 'coronavirus',
            'influenza', 'h1n1', 'h5n1', 'sars', 'mers', 'ebola'
        ]
        
        found_viruses = []
        for pattern in virus_patterns:
            if pattern in query_lower:
                found_viruses.append(pattern)
        
        return found_viruses
    
    def extract_protein_sequences(self, query: str) -> List[str]:
        """Extract protein sequences from query"""
        import re
        
        # Look for amino acid sequences (basic pattern)
        protein_pattern = r'[ACDEFGHIKLMNPQRSTVWY]{10,}'
        sequences = re.findall(protein_pattern, query.upper())
        
        return sequences
    
    def extract_analysis_types(self, query_lower: str) -> List[str]:
        """Extract analysis types from query"""
        analysis_patterns = [
            'pssm', 'phylogenetic', 'alignment', 'annotation', 'structure',
            'function', 'similarity', 'comparison', 'prediction', 'modeling'
        ]
        
        found_types = []
        for pattern in analysis_patterns:
            if pattern in query_lower:
                found_types.append(pattern)
        
        return found_types
    
    def calculate_complexity_score(self, query: str, intent: IntentClassification, 
                                 domain: DomainClassification, entities: Dict[str, Any]) -> float:
        """Calculate query complexity score"""
        try:
            score = 0.5  # Base score
            
            # Adjust based on query length
            query_length = len(query.split())
            if query_length > 20:
                score += 0.2
            elif query_length > 10:
                score += 0.1
            
            # Adjust based on intent complexity
            complex_intents = [IntentType.ANALYSIS_REQUEST, IntentType.COMPARISON_REQUEST]
            if intent.intent_type in complex_intents:
                score += 0.2
            
            # Adjust based on domain complexity
            complex_domains = [DomainType.BIOINFORMATICS, DomainType.PROTEIN_ANALYSIS]
            if domain.domain_type in complex_domains:
                score += 0.2
            
            # Adjust based on entities
            if entities:
                entity_count = sum(len(v) if isinstance(v, list) else 1 for v in entities.values())
                score += min(entity_count * 0.1, 0.3)
            
            # Clamp to valid range
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Complexity calculation failed: {e}")
            return 0.5
    
    def fallback_intent_classification(self, query: str) -> IntentClassification:
        """Fallback intent classification using simple rules"""
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        if any(word in query_lower for word in ['analyze', 'analysis', 'examine', 'study']):
            intent_type = IntentType.ANALYSIS_REQUEST
            confidence = 0.6
        elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
            intent_type = IntentType.COMPARISON_REQUEST
            confidence = 0.6
        elif any(word in query_lower for word in ['what is', 'explain', 'describe', 'tell me']):
            intent_type = IntentType.EXPLANATION_REQUEST
            confidence = 0.7
        elif any(word in query_lower for word in ['how to', 'procedure', 'method', 'steps']):
            intent_type = IntentType.PROCEDURE_REQUEST
            confidence = 0.6
        else:
            intent_type = IntentType.GENERAL_CONVERSATION
            confidence = 0.4
        
        return IntentClassification(
            intent_type=intent_type,
            confidence=confidence,
            keywords=[],
            classification_method='fallback_rules',
            metadata={'fallback': True}
        )
    
    def fallback_domain_classification(self, query: str) -> DomainClassification:
        """Fallback domain classification using simple rules"""
        query_lower = query.lower()
        
        # Simple keyword-based domain detection
        if any(word in query_lower for word in ['virus', 'viral', 'pathogen', 'infection']):
            domain_type = DomainType.VIROLOGY
            confidence = 0.7
        elif any(word in query_lower for word in ['protein', 'amino acid', 'sequence', 'structure']):
            domain_type = DomainType.PROTEIN_ANALYSIS
            confidence = 0.7
        elif any(word in query_lower for word in ['genome', 'dna', 'rna', 'gene', 'genetic']):
            domain_type = DomainType.GENOMICS
            confidence = 0.7
        elif any(word in query_lower for word in ['bioinformatics', 'computational', 'algorithm']):
            domain_type = DomainType.BIOINFORMATICS
            confidence = 0.6
        else:
            domain_type = DomainType.GENERAL_SCIENCE
            confidence = 0.4
        
        return DomainClassification(
            domain_type=domain_type,
            confidence=confidence,
            indicators=[],
            classification_method='fallback_rules',
            metadata={'fallback': True}
        )
    
    def create_fallback_analysis(self, request: ChatRequest) -> RequestAnalysis:
        """Create fallback analysis when main analysis fails"""
        return RequestAnalysis(
            request_id=request.request_id or str(uuid.uuid4()),
            original_query=request.query,
            intent_classification=self.fallback_intent_classification(request.query),
            domain_classification=self.fallback_domain_classification(request.query),
            extracted_entities={},
            complexity_score=0.5,
            analysis_timestamp=datetime.now(),
            analyzer_metadata={
                'analysis_method': 'fallback',
                'error_occurred': True
            }
        )
    
    def get_cached_analysis(self, query: str) -> Optional[RequestAnalysis]:
        """Get cached analysis if available and valid"""
        cache_key = hash(query.strip().lower())
        cached = self.analysis_cache.get(str(cache_key))
        
        if cached:
            # Check if cache is still valid
            cache_age = (datetime.now() - cached.analysis_timestamp).total_seconds()
            cache_ttl = self.config.performance_config.get('cache_ttl_seconds', 3600)
            
            if cache_age < cache_ttl:
                return cached
            else:
                # Remove expired cache entry
                del self.analysis_cache[str(cache_key)]
        
        return None
    
    def cache_analysis(self, query: str, analysis: RequestAnalysis) -> None:
        """Cache analysis result"""
        cache_key = str(hash(query.strip().lower()))
        self.analysis_cache[cache_key] = analysis
        
        # Simple cache size management
        if len(self.analysis_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.analysis_cache.keys(), 
                               key=lambda k: self.analysis_cache[k].analysis_timestamp)[:200]
            for key in oldest_keys:
                del self.analysis_cache[key]
    
    async def get_health_status(self) -> str:
        """Get analyzer health status"""
        try:
            # Test basic functionality
            test_request = ChatRequest(query="test", options={})
            await self.analyze_request(test_request)
            return "healthy"
        except Exception as e:
            logger.error(f"❌ Analyzer health check failed: {e}")
            return "unhealthy" 