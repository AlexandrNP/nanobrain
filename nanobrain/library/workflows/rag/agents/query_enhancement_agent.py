#!/usr/bin/env python3
"""
Query Enhancement Agent for RAG systems.
The ONLY LLM-based component in the RAG pipeline.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from nanobrain.core.logging_system import get_logger
from nanobrain.core.agent import SimpleAgent

logger = get_logger(__name__)


class QueryEnhancementAgent(SimpleAgent):
    """
    LLM-based query enhancement agent for RAG systems.
    
    This is the single component that uses LLM processing for natural language
    understanding and query optimization. All other RAG components are deterministic.
    
    Key Features:
    - Template-based prompts loaded from YAML configuration
    - No hardcoded prompts or templates
    - Graceful fallback system with multiple levels
    - JSON-structured output for downstream processing
    """
    
    COMPONENT_TYPE = "query_enhancement_agent"
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize QueryEnhancementAgent from configuration using framework pattern."""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)

        # Store config for backward compatibility
        self.config = config
        self.initialized = False



        # Initialize template system from configuration
        self.prompt_template = self._load_template_from_config()

        # Initialize configuration attributes for runtime access
        self.max_query_length = getattr(self.config, 'max_query_length', 500)
        self.preserve_original_intent = getattr(self.config, 'preserve_original_intent', True)
        self.template_variables = getattr(self.config, 'template_variables', {})

        logger.info(f"🧠 QueryEnhancementAgent initialized with template from configuration")

    async def initialize(self):
        """Initialize the agent."""
        # Call base class initialization to set up LLM client
        await super().initialize()

        # Agent-specific initialization
        self.initialized = True
        logger.info(f"✅ QueryEnhancementAgent initialized: {self.name}")


    def _load_template_from_config(self) -> 'Template':
        """Load query enhancement prompt template from YAML configuration."""
        from jinja2 import Template

        # Get main template from configuration
        main_template = getattr(self.config, 'prompt_template', None)
        
        if main_template:
            try:
                return Template(main_template)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load main template from config: {e}")
                return self._load_fallback_template_from_config()
        else:
            logger.warning("⚠️ No main template found in config, using fallback")
            return self._load_fallback_template_from_config()
    
    def _load_fallback_template_from_config(self) -> 'Template':
        """Load fallback template from YAML configuration."""
        from jinja2 import Template

        # Get fallback template from configuration
        fallback_template = getattr(self.config, 'fallback_template', None)
        
        if fallback_template:
            try:
                return Template(fallback_template)
            except Exception as e:
                logger.error(f"❌ Failed to load fallback template from config: {e}")
                raise ValueError("No valid templates found in configuration. Please provide 'prompt_template' or 'fallback_template' in agent configuration.")
        else:
            raise ValueError("No templates found in configuration. Please provide 'prompt_template' and 'fallback_template' in agent configuration.")
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process and enhance user queries using LLM reasoning.
        
        Args:
            input_text: Original user query
            **kwargs: Additional context including conversation history
            
        Returns:
            Enhanced query with optimization metadata as JSON string
        """
        try:
            # Extract context
            context = kwargs.get('context', {})
            conversation_history = kwargs.get('conversation_history', [])
            domain_context = kwargs.get('domain_context', '')
            
            # Build enhancement prompt using template
            enhancement_prompt = self._build_enhancement_prompt(
                input_text, context, conversation_history, domain_context
            )
            
            # Call parent's LLM processing
            enhanced_result = await super().process(enhancement_prompt, **kwargs)
            
            # Parse and validate enhancement
            parsed_enhancement = self._parse_enhancement_result(enhanced_result, input_text)
            
            logger.info(f"✅ Query enhanced: '{input_text[:50]}...' -> '{parsed_enhancement['enhanced_query'][:50]}...'")
            
            return self._format_enhancement_output(parsed_enhancement)
            
        except Exception as e:
            logger.error(f"❌ Query enhancement failed: {e}")
            # Graceful fallback to original query
            return self._create_fallback_enhancement(input_text)
    
    def _build_enhancement_prompt(self, query: str, context: Dict[str, Any],
                                history: List[Dict], domain: str) -> str:
        """Build LLM prompt for query enhancement using template system."""
        try:
            # Prepare template variables
            template_vars = {
                'original_query': query,
                'max_query_length': getattr(self.config, 'max_query_length', 500),
                'domain_context': domain or 'General',
                'enhancement_strategies': getattr(self.config, 'enhancement_strategies', [
                    'synonym_expansion', 'context_integration', 'domain_optimization'
                ]),
                'preserve_original_intent': getattr(self.config, 'preserve_original_intent', True),
                'conversation_history': history,
                'additional_context': context,
                **getattr(self.config, 'template_variables', {})  # Include any additional template variables from config
            }
            
            # Render prompt using template
            return self.prompt_template.render(**template_vars)
            
        except Exception as e:
            logger.error(f"❌ Template rendering failed: {e}")
            # Try fallback template
            return self._create_fallback_prompt(query, context, history, domain)
    
    def _create_fallback_prompt(self, query: str, context: Dict[str, Any],
                               history: List[Dict], domain: str) -> str:
        """Create fallback prompt using fallback template from configuration."""
        try:
            # Load fallback template from config
            fallback_template = self._load_fallback_template_from_config()
            
            # Prepare fallback template variables
            fallback_vars = {
                'original_query': query,
                'domain_context': domain or 'General',
                'max_query_length': self.max_query_length,
                'conversation_count': len(history),
                'additional_context': context,
                'conversation_history': history,
                **self.template_variables  # Include configured template variables
            }
            
            # Render fallback prompt using template
            return fallback_template.render(**fallback_vars)
            
        except Exception as e:
            logger.error(f"❌ Fallback template rendering failed: {e}")
            # Last resort: return emergency prompt
            return self._create_emergency_prompt(query, domain)
    
    def _create_emergency_prompt(self, query: str, domain: str) -> str:
        """Create emergency prompt when all template systems fail."""
        # Use only configuration-driven content, no hardcoded prompts
        emergency_instruction = getattr(self.config, 'emergency_instruction', None)
        
        if not emergency_instruction:
            raise ValueError("No emergency_instruction found in configuration. All template systems failed and no fallback instruction provided.")
        
        return f"{emergency_instruction}: {query} (Domain: {domain})"
    
    def _parse_enhancement_result(self, result: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM enhancement result with fallback handling."""
        try:
            # Try to parse JSON response
            if result.strip().startswith('{'):
                parsed = json.loads(result)
                return {
                    'enhanced_query': parsed.get('enhanced_query', original_query),
                    'search_terms': parsed.get('search_terms', []),
                    'context_additions': parsed.get('context_additions', ''),
                    'enhancement_rationale': parsed.get('enhancement_rationale', ''),
                    'confidence_score': parsed.get('confidence_score', 0.8)
                }
            else:
                # Fallback: treat entire response as enhanced query
                return {
                    'enhanced_query': result.strip()[:self.max_query_length],
                    'search_terms': result.split()[:10],
                    'context_additions': '',
                    'enhancement_rationale': 'LLM provided text enhancement',
                    'confidence_score': 0.7
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse enhancement result: {e}")
            return self._create_fallback_enhancement(original_query)
    
    def _create_fallback_enhancement(self, original_query: str) -> str:
        """Create fallback enhancement when LLM processing fails."""
        fallback_result = {
            'enhanced_query': original_query,
            'search_terms': original_query.split(),
            'context_additions': '',
            'enhancement_rationale': 'Fallback: using original query',
            'confidence_score': 0.5
        }
        return json.dumps(fallback_result, indent=2)
    
    def _format_enhancement_output(self, enhancement: Dict[str, Any]) -> str:
        """Format enhancement result for downstream processing."""
        return json.dumps(enhancement, indent=2)
