"""
Query Classification Step for Chatbot Viral Integration Workflow

SINGLE RESPONSIBILITY: Extract virus species using LLM agent (NO hardcoded patterns)
"""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.library.agents.specialized.simple_specialized_agent import SimpleSpecializedAgent


class QueryClassificationStep(Step):
    """
    Query Classification Step implementing LLM-based virus species extraction
    
    SINGLE RESPONSIBILITY: Extract virus species using LLM agent
    NO hardcoded patterns - all extraction via LLM with configurable prompts
    """
    
    COMPONENT_TYPE = "step"
    REQUIRED_CONFIG_FIELDS = ['name']
    
    def __init__(self, config: StepConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.extraction_agent = None
        
    @classmethod
    def from_config(cls, config: StepConfig, **kwargs) -> 'QueryClassificationStep':
        """Create QueryClassificationStep from configuration"""
        return cls(config, **kwargs)
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """Initialize QueryClassificationStep with LLM agent"""
        super()._init_from_config(config, component_config, dependencies)
        self._create_extraction_agent()
    
    def _create_extraction_agent(self) -> None:
        """Create LLM agent for virus species extraction"""
        try:
            # Get agent configuration from step config
            agent_config_dict = getattr(self.config, 'extraction_agent', {})
            
            # Create SimpleSpecializedAgent configuration
            agent_config = StepConfig(
                name="virus_extraction_agent",
                model=agent_config_dict.get('model', 'gpt-4'),
                temperature=agent_config_dict.get('temperature', 0.1),
                max_tokens=agent_config_dict.get('max_tokens', 100),
                timeout=agent_config_dict.get('timeout', 30),
                **agent_config_dict
            )
            
            # Create agent using from_config pattern
            self.extraction_agent = SimpleSpecializedAgent.from_config(agent_config)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("✅ Created LLM agent for virus species extraction")
                
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create extraction agent: {e}")
            raise
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process method implementing LLM-based virus species extraction
        
        Args:
            input_data: Must contain 'user_query' key
            
        Returns:
            Dict with routing decision and extracted virus species
        """
        if hasattr(self, 'nb_logger') and self.nb_logger:
            self.nb_logger.info("🔄 Processing query classification step with LLM-based extraction")
        
        query = input_data.get('user_query', '')
        if not query:
            return {
                'routing_decision': 'conversational_response',
                'error': 'No user query provided'
            }
        
        # Extract virus species using LLM agent - NO hardcoded patterns
        extraction_result = await self._extract_virus_species_llm(query)
        
        # Route based on extraction success
        if extraction_result.get('virus_species'):
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"✅ Extracted virus species: {extraction_result['virus_species']}")
            
            return {
                'routing_decision': 'virus_name_resolution',
                'extracted_virus_species': extraction_result['virus_species'],
                'confidence': extraction_result.get('confidence', 0.0),
                'user_query': query,
                'extraction_reasoning': extraction_result.get('reasoning', '')
            }
        else:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("ℹ️ No virus species detected, routing to conversational response")
            
            return {
                'routing_decision': 'conversational_response',
                'user_query': query,
                'extraction_reasoning': extraction_result.get('reasoning', 'No virus species detected')
            }
    
    async def _extract_virus_species_llm(self, query: str) -> Dict[str, Any]:
        """Extract virus species using LLM agent with configurable prompts"""
        try:
            # Get prompt template from configuration (NO hardcoded patterns)
            prompt_template = getattr(self.config, 'virus_extraction_prompt', 
                '''Analyze the following user query and extract any virus species mentioned:

Query: "{user_query}"

Extract virus species names if present. Return JSON format:
{{
  "virus_species": "exact virus species name or null if none found",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Examples of virus species: Chikungunya virus, Eastern equine encephalitis virus, Zika virus

Return only the JSON response.''')
            
            formatted_prompt = prompt_template.format(user_query=query)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🤖 Sending virus extraction prompt to LLM")
            
            # Process using LLM agent
            response = await self.extraction_agent.process({
                'prompt': formatted_prompt,
                'expected_format': 'json',
                'user_query': query
            })
            
            # Parse LLM response
            return self._parse_extraction_response(response)
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ LLM extraction failed: {e}")
            
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'LLM extraction failed: {str(e)}'
            }
    
    def _parse_extraction_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response and extract virus species information"""
        try:
            # Handle different response formats
            if isinstance(response, dict):
                if 'response' in response:
                    # Response is nested
                    response_text = response['response']
                elif 'content' in response:
                    # Response has content key
                    response_text = response['content']
                else:
                    # Response is direct
                    response_text = str(response)
            else:
                response_text = str(response)
            
            # Try to parse as JSON
            import json
            try:
                if response_text.strip().startswith('{'):
                    parsed = json.loads(response_text)
                    
                    return {
                        'virus_species': parsed.get('virus_species'),
                        'confidence': float(parsed.get('confidence', 0.0)),
                        'reasoning': parsed.get('reasoning', '')
                    }
                else:
                    # Response is not JSON, try to extract manually
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.warning(f"⚠️ Non-JSON response from LLM: {response_text[:100]}...")
                    
                    return {
                        'virus_species': None,
                        'confidence': 0.0,
                        'reasoning': 'Could not parse LLM response as JSON'
                    }
                    
            except json.JSONDecodeError as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.error(f"❌ JSON parsing failed: {e}")
                
                return {
                    'virus_species': None,
                    'confidence': 0.0,
                    'reasoning': f'JSON parsing failed: {str(e)}'
                }
                
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Response parsing failed: {e}")
            
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Response parsing failed: {str(e)}'
            } 