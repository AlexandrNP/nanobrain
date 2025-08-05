#!/usr/bin/env python3
"""
Intent Classifier for NanoBrain Framework
Classifies user intent from natural language requests using configurable methods.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import logging
import re
from typing import Dict, Any, Optional, List, Set, Pattern

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.universal_models import IntentClassification, IntentType
from pydantic import Field

# Intent classifier logger
logger = logging.getLogger(__name__)


class IntentClassifierConfig(ConfigBase):
    """Configuration for intent classifier"""
    
    # Classification method
    method: str = Field(
        default="keyword_based",
        description="Classification method to use"
    )
    
    # Intent patterns for classification (component expects dict with keywords sub-key)
    intent_patterns: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: {
            'question': {
                'keywords': ['what', 'how', 'when', 'where', 'why', 'explain', 'describe']
            },
            'analysis_request': {
                'keywords': ['analyze', 'examine', 'study', 'investigate', 'process']
            },
            'data_request': {
                'keywords': ['show', 'display', 'list', 'find', 'search']
            },
            'workflow_request': {
                'keywords': ['run', 'execute', 'start', 'perform']
            }
        },
        description="Intent patterns for classification with keywords structure"
    )
    
    # Advanced pattern configuration (component expects this as dict)
    advanced_patterns: Dict[str, Any] = Field(
        default_factory=lambda: {
            'use_regex': True,
            'case_sensitive': False,
            'word_boundaries': True,
            'enable_stemming': False,
            'enable_synonyms': False
        },
        description="Advanced pattern matching configuration"
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
    
    # Method-specific configurations
    keyword_based_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'patterns': {
                'question': ['what', 'how', 'when', 'where', 'why', 'explain', 'describe'],
                'analysis_request': ['analyze', 'examine', 'study', 'investigate', 'process'],
                'data_request': ['show', 'display', 'list', 'find', 'search'],
                'workflow_request': ['run', 'execute', 'start', 'perform']
            },
            'case_sensitive': False
        },
        description="Keyword-based classification configuration"
    )
    
    # Rule-based configuration
    rule_based_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'rules_file': None,
            'custom_rules': [],
            'rule_priority': 'first_match'
        },
        description="Rule-based classification configuration"
    )
    
    # Semantic configuration
    semantic_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'enable_preprocessing': True
        },
        description="Semantic classification configuration"
    )
    
    # Performance settings
    performance_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_caching': True,
            'cache_ttl_seconds': 1800,
            'batch_processing': False
        },
        description="Performance optimization settings"
    )


class IntentClassifier(FromConfigBase):
    """Configurable intent classification for natural language requests"""
    
    def __init__(self):
        """Initialize intent classifier - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return IntentClassifierConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize classifier from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[IntentClassifierConfig] = None
        self.compiled_patterns: Dict[str, List[Pattern]] = {}
        
        logger.info("🎯 Initializing Intent Classifier")
        self.config = config
        
        # Compile patterns for efficiency
        self.compile_intent_patterns()
        
        logger.info(f"✅ Intent Classifier initialized with {self.config.method} method")
    
    def compile_intent_patterns(self) -> None:
        """Compile regex patterns for efficient matching"""
        try:
            if not self.config.advanced_patterns.get('use_regex', True):
                return
            
            flags = 0 if self.config.advanced_patterns.get('case_sensitive', False) else re.IGNORECASE
            
            for intent_name, pattern_config in self.config.intent_patterns.items():
                compiled_patterns = []
                
                for keyword in pattern_config.get('keywords', []):
                    if self.config.advanced_patterns.get('word_boundaries', True):
                        # Add word boundaries for more precise matching
                        pattern = r'\b' + re.escape(keyword) + r'\b'
                    else:
                        pattern = re.escape(keyword)
                    
                    compiled_patterns.append(re.compile(pattern, flags))
                
                self.compiled_patterns[intent_name] = compiled_patterns
            
            logger.debug(f"✅ Compiled patterns for {len(self.compiled_patterns)} intent types")
            
        except Exception as e:
            logger.error(f"❌ Pattern compilation failed: {e}")
            raise
    
    async def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify intent using configured method.
        
        Args:
            query: Natural language query to classify
            
        Returns:
            IntentClassification with detected intent and confidence
        """
        try:
            if self.config.method == 'keyword_based':
                return self.classify_intent_keyword_based(query)
            elif self.config.method == 'rule_based':
                return self.classify_intent_rule_based(query)
            elif self.config.method == 'hybrid':
                return self.classify_intent_hybrid(query)
            else:
                logger.warning(f"⚠️ Unknown classification method: {self.config.method}")
                return self.classify_intent_keyword_based(query)
                
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def classify_intent_keyword_based(self, query: str) -> IntentClassification:
        """Classify intent using keyword-based approach"""
        try:
            intent_scores = {}
            matched_keywords = []
            
            # Score each intent type based on keyword matches
            for intent_name, pattern_config in self.config.intent_patterns.items():
                score = 0.0
                intent_keywords = []
                
                if self.compiled_patterns and intent_name in self.compiled_patterns:
                    # Use compiled regex patterns
                    for pattern in self.compiled_patterns[intent_name]:
                        matches = pattern.findall(query)
                        if matches:
                            score += len(matches) * pattern_config.get('weight', 1.0)
                            intent_keywords.extend(matches)
                else:
                    # Fallback to simple string matching
                    query_lower = query.lower()
                    for keyword in pattern_config.get('keywords', []):
                        if keyword.lower() in query_lower:
                            score += pattern_config.get('weight', 1.0)
                            intent_keywords.append(keyword)
                
                if score > 0:
                    intent_scores[intent_name] = score
                    matched_keywords.extend(intent_keywords)
            
            # Determine best intent
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                raw_confidence = intent_scores[best_intent]
                
                # Normalize confidence score
                confidence = min(1.0, raw_confidence / 3.0)  # Scale down for realistic confidence
                
                # Map intent name to enum
                intent_type = self.map_intent_name_to_enum(best_intent)
                
                return IntentClassification(
                    intent_type=intent_type,
                    confidence=confidence,
                    keywords=list(set(matched_keywords)),
                    classification_method='keyword_based',
                    metadata={
                        'all_scores': intent_scores,
                        'best_score': raw_confidence
                    }
                )
            else:
                # No intent detected
                return IntentClassification(
                    intent_type=IntentType.GENERAL_CONVERSATION,
                    confidence=0.3,
                    keywords=[],
                    classification_method='keyword_based',
                    metadata={'no_patterns_matched': True}
                )
                
        except Exception as e:
            logger.error(f"❌ Keyword-based classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def classify_intent_rule_based(self, query: str) -> IntentClassification:
        """Classify intent using rule-based approach with more complex logic"""
        try:
            query_lower = query.lower().strip()
            confidence = 0.5
            keywords = []
            
            # Question patterns
            if query_lower.startswith(('what is', 'what are', 'what does')):
                intent_type = IntentType.INFORMATION_REQUEST
                confidence = 0.8
                keywords = ['question_pattern']
            elif query_lower.startswith(('how does', 'how do', 'how can')):
                intent_type = IntentType.EXPLANATION_REQUEST
                confidence = 0.8
                keywords = ['how_question']
            elif query_lower.startswith(('why does', 'why do', 'why is')):
                intent_type = IntentType.EXPLANATION_REQUEST
                confidence = 0.8
                keywords = ['why_question']
            
            # Command patterns
            elif any(word in query_lower for word in ['analyze', 'examine', 'study']):
                intent_type = IntentType.ANALYSIS_REQUEST
                confidence = 0.7
                keywords = ['analysis_command']
            elif any(word in query_lower for word in ['compare', 'versus', 'vs']):
                intent_type = IntentType.COMPARISON_REQUEST
                confidence = 0.7
                keywords = ['comparison_command']
            
            # Procedure patterns
            elif 'how to' in query_lower or 'steps' in query_lower:
                intent_type = IntentType.PROCEDURE_REQUEST
                confidence = 0.7
                keywords = ['procedure_pattern']
            
            # Default to conversation
            else:
                intent_type = IntentType.GENERAL_CONVERSATION
                confidence = 0.4
                keywords = ['default']
            
            return IntentClassification(
                intent_type=intent_type,
                confidence=confidence,
                keywords=keywords,
                classification_method='rule_based',
                metadata={'query_pattern': 'analyzed'}
            )
            
        except Exception as e:
            logger.error(f"❌ Rule-based classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def classify_intent_hybrid(self, query: str) -> IntentClassification:
        """Classify intent using hybrid approach combining keyword and rule-based methods"""
        try:
            # Get classifications from both methods
            keyword_result = self.classify_intent_keyword_based(query)
            rule_result = self.classify_intent_rule_based(query)
            
            # Combine results with weighted average
            keyword_weight = 0.7
            rule_weight = 0.3
            
            # If both methods agree, increase confidence
            if keyword_result.intent_type == rule_result.intent_type:
                combined_confidence = min(1.0, 
                    (keyword_result.confidence * keyword_weight + 
                     rule_result.confidence * rule_weight) * 1.2  # Boost for agreement
                )
                intent_type = keyword_result.intent_type
                combined_keywords = list(set(keyword_result.keywords + rule_result.keywords))
            else:
                # Use the result with higher confidence
                if keyword_result.confidence >= rule_result.confidence:
                    intent_type = keyword_result.intent_type
                    combined_confidence = keyword_result.confidence * 0.9  # Slight penalty for disagreement
                    combined_keywords = keyword_result.keywords
                else:
                    intent_type = rule_result.intent_type
                    combined_confidence = rule_result.confidence * 0.9
                    combined_keywords = rule_result.keywords
            
            return IntentClassification(
                intent_type=intent_type,
                confidence=combined_confidence,
                keywords=combined_keywords,
                classification_method='hybrid',
                metadata={
                    'keyword_result': keyword_result.dict(),
                    'rule_result': rule_result.dict(),
                    'methods_agreed': keyword_result.intent_type == rule_result.intent_type
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Hybrid classification failed: {e}")
            return self.create_fallback_classification(query)
    
    def map_intent_name_to_enum(self, intent_name: str) -> IntentType:
        """Map intent name to IntentType enum"""
        mapping = {
            'information_request': IntentType.INFORMATION_REQUEST,
            'analysis_request': IntentType.ANALYSIS_REQUEST,
            'comparison_request': IntentType.COMPARISON_REQUEST,
            'explanation_request': IntentType.EXPLANATION_REQUEST,
            'procedure_request': IntentType.PROCEDURE_REQUEST,
            'general_conversation': IntentType.GENERAL_CONVERSATION
        }
        
        return mapping.get(intent_name, IntentType.UNKNOWN)
    
    def create_fallback_classification(self, query: str) -> IntentClassification:
        """Create fallback classification when main classification fails"""
        return IntentClassification(
            intent_type=IntentType.GENERAL_CONVERSATION,
            confidence=0.2,
            keywords=[],
            classification_method='fallback',
            metadata={'error_occurred': True}
        )
    
    def get_supported_intents(self) -> List[IntentType]:
        """Get list of supported intent types"""
        return [
            IntentType.INFORMATION_REQUEST,
            IntentType.ANALYSIS_REQUEST,
            IntentType.COMPARISON_REQUEST,
            IntentType.EXPLANATION_REQUEST,
            IntentType.PROCEDURE_REQUEST,
            IntentType.GENERAL_CONVERSATION
        ]
    
    def update_intent_patterns(self, new_patterns: Dict[str, Dict[str, Any]]) -> None:
        """Update intent patterns dynamically"""
        try:
            self.config.intent_patterns.update(new_patterns)
            self.compile_intent_patterns()
            logger.info(f"✅ Updated intent patterns for {len(new_patterns)} intents")
        except Exception as e:
            logger.error(f"❌ Failed to update intent patterns: {e}")
            raise 