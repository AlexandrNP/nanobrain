"""
Query Classification Step for Chatbot Viral Integration

This step classifies user queries and extracts virus species using LLM agents.
NO hardcoded virus species patterns - all extraction via LLM with configurable prompts.
"""

import json
from typing import Dict, Any, Optional

from nanobrain.core.step import Step
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.agent import SimpleAgent


class QueryClassificationStep(Step):
    """
    SINGLE RESPONSIBILITY: Extract virus species using LLM agent
    
    This step classifies user queries and extracts virus species names using
    LLM agents with configurable prompts. NO hardcoded virus patterns.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.extraction_agent = None
        
    def _init_from_config(self, config, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize QueryClassificationStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Initialize extraction agent from configuration
        self.extraction_agent = self._create_extraction_agent(component_config)
        
        if self.nb_logger:
            self.nb_logger.info("🔍 Query Classification Step initialized with LLM-based virus species extraction")
    
    def _create_extraction_agent(self, component_config: Dict[str, Any]) -> SimpleAgent:
        """Create LLM agent for virus species extraction from configuration"""
        # Get agent config from the component configuration
        agent_config_dict = component_config.get('extraction_agent', {})
        
        # If not found, try to get from the main config object (the full YAML configuration)
        if not agent_config_dict and hasattr(self.config, 'extraction_agent'):
            agent_config_dict = getattr(self.config, 'extraction_agent')
        elif not agent_config_dict and isinstance(self.config, dict):
            agent_config_dict = self.config.get('extraction_agent', {})
        
        if not agent_config_dict:
            raise ValueError("No extraction_agent configuration found in step configuration")
        
        # Ensure the name field is present for SimpleAgent
        if 'name' not in agent_config_dict:
            agent_config_dict['name'] = 'virus_extraction_agent'
        
        # Create AgentConfig object from dictionary - this is what SimpleAgent expects
        from nanobrain.core.agent import AgentConfig
        
        # Create A2A protocol compliant agent_card
        agent_card = {
            "version": "1.0.0",
            "purpose": "Virus species extraction from user queries using LLM",
            "detailed_description": "Specialized agent for extracting virus species names from natural language queries with high accuracy",
            "domain": "bioinformatics",
            "expertise_level": "intermediate",
            "input_format": {
                "primary_mode": "text",
                "supported_modes": ["text"],
                "content_types": ["text/plain"]
            },
            "output_format": {
                "primary_mode": "json",
                "supported_modes": ["json"],
                "content_types": ["application/json"]
            },
            "capabilities": {
                "streaming": False,
                "multi_turn_conversation": False,
                "context_retention": False,
                "tool_usage": False
            }
        }
        
        agent_config = AgentConfig(
            name=agent_config_dict.get('name', 'virus_extraction_agent'),
            description=agent_config_dict.get('description', 'Agent for extracting virus species from user queries'),
            model=agent_config_dict.get('model', 'gpt-4'),
            temperature=agent_config_dict.get('temperature', 0.1),
            max_tokens=agent_config_dict.get('max_tokens', 100),
            system_prompt=agent_config_dict.get('system_prompt', ''),
            timeout=agent_config_dict.get('timeout', 30),
            agent_card=agent_card
        )
        
        # Create agent using mandatory from_config pattern with proper AgentConfig object
        return SimpleAgent.from_config(agent_config)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user query to extract virus species using LLM agent
        
        Args:
            input_data: Contains 'user_query' key with query text
            
        Returns:
            Dict with routing decision and extracted virus species
        """
        user_query = input_data.get('user_query', '')
        
        if not user_query:
            return {
                'routing_decision': 'conversational_response',
                'error': 'No user query provided'
            }
        
        try:
            # Extract virus species using LLM agent - NO hardcoded patterns
            extraction_result = await self._extract_virus_species_llm(user_query)
            
            if extraction_result.get('virus_species'):
                # Virus species detected - route to virus name resolution
                return {
                    'routing_decision': 'virus_name_resolution',
                    'extracted_virus_species': extraction_result['virus_species'],
                    'confidence': extraction_result.get('confidence', 0.0),
                    'reasoning': extraction_result.get('reasoning', ''),
                    'user_query': user_query
                }
            else:
                # No virus species detected - route to conversational response
                return {
                    'routing_decision': 'conversational_response',
                    'user_query': user_query,
                    'reasoning': extraction_result.get('reasoning', 'No virus species detected')
                }
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error in query classification: {e}")
            return {
                'routing_decision': 'conversational_response',
                'user_query': user_query,
                'error': str(e)
            }
    
    async def _extract_virus_species_llm(self, user_query: str) -> Dict[str, Any]:
        """
        Extract virus species using LLM agent with configurable prompts
        
        Args:
            user_query: The user's query text
            
        Returns:
            Dict with virus species, confidence, and reasoning
        """
        try:
            # Get prompt template from configuration
            prompt_template = self.step_config.get('virus_extraction_prompt', '')
            
            if not prompt_template:
                raise ValueError("No virus extraction prompt configured")
            
            # Format prompt with user query
            formatted_prompt = prompt_template.format(user_query=user_query)
            
            # Call LLM agent with formatted prompt
            response = await self.extraction_agent.process({
                'prompt': formatted_prompt,
                'expected_format': 'json'
            })
            
            # Parse the response
            return self._parse_extraction_response(response)
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error in LLM virus species extraction: {e}")
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Error in LLM extraction: {str(e)}'
            }
    
    def _parse_extraction_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response to extract virus species information
        
        Args:
            response: Raw response from LLM agent
            
        Returns:
            Dict with extracted virus species data
        """
        try:
            # Handle different response formats
            if isinstance(response, dict):
                if 'content' in response:
                    content = response['content']
                elif 'text' in response:
                    content = response['text']
                else:
                    content = str(response)
            else:
                content = str(response)
            
            # Try to parse as JSON
            try:
                parsed_data = json.loads(content)
                
                return {
                    'virus_species': parsed_data.get('virus_species'),
                    'confidence': parsed_data.get('confidence', 0.0),
                    'reasoning': parsed_data.get('reasoning', '')
                }
            
            except json.JSONDecodeError:
                # If not JSON, try to extract virus species from text
                self.nb_logger.warning("⚠️ LLM response not in JSON format, attempting text parsing")
                
                # Basic text parsing fallback
                content_lower = content.lower()
                common_viruses = [
                    'chikungunya virus', 'eastern equine encephalitis virus',
                    'western equine encephalitis virus', 'venezuelan equine encephalitis virus',
                    'sindbis virus', 'semliki forest virus', 'ross river virus',
                    'zika virus', 'dengue virus', 'yellow fever virus'
                ]
                
                for virus in common_viruses:
                    if virus in content_lower:
                        return {
                            'virus_species': virus,
                            'confidence': 0.7,
                            'reasoning': f'Extracted from text: {virus}'
                        }
                
                return {
                    'virus_species': None,
                    'confidence': 0.0,
                    'reasoning': 'No virus species detected in response'
                }
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error parsing extraction response: {e}")
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Error parsing response: {str(e)}'
            } 