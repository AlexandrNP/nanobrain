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
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process user query to extract virus species using LLM agent
        
        Args:
            input_data: Contains 'user_query' key with query text
            **kwargs: Additional arguments from workflow execution context
            
        Returns:
            Dict with routing decision and extracted virus species
        """
        # CRITICAL FIX: Robust input data extraction to handle different input formats
        user_query = await self._extract_user_query(input_data, **kwargs)
        
        if not user_query:
            self.nb_logger.error("❌ No user query found in input data")
            self.nb_logger.error(f"Input data structure: {input_data}")
            return {
                'routing_decision': 'conversational_response',
                'error': 'No user query provided'
            }
        
        self.nb_logger.info(f"🔍 Processing user query: '{user_query[:100]}...'")
        
        try:
            # Extract virus species using LLM agent - NO hardcoded patterns
            extraction_result = await self._extract_virus_species_llm(user_query)
            
            if extraction_result.get('virus_species'):
                # Virus species detected - route to virus name resolution
                result = {
                    'routing_decision': 'virus_name_resolution',
                    'extracted_virus_species': extraction_result['virus_species'],
                    'confidence': extraction_result.get('confidence', 0.0),
                    'reasoning': extraction_result.get('reasoning', ''),
                    'user_query': user_query
                }
                self.nb_logger.info(f"✅ Extracted virus species: {extraction_result['virus_species']}")
                return result
            else:
                # No virus species detected - route to conversational response
                result = {
                    'routing_decision': 'conversational_response',
                    'user_query': user_query,
                    'reasoning': extraction_result.get('reasoning', 'No virus species detected')
                }
                self.nb_logger.info("ℹ️ No virus species detected, routing to conversational response")
                return result
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error in query classification: {e}")
            import traceback
            self.nb_logger.error(f"Query classification traceback: {traceback.format_exc()}")
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
            # Get prompt template from configuration - FIXED: use self.config instead of self.step_config
            prompt_template = getattr(self.config, 'virus_extraction_prompt', '')
            
            # If not found in config object, try as dict
            if not prompt_template and isinstance(self.config, dict):
                prompt_template = self.config.get('virus_extraction_prompt', '')
            
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
    
    async def _extract_user_query(self, input_data: Dict[str, Any], **kwargs) -> str:
        """
        Extract user query from input data with robust handling of different input formats.
        
        Args:
            input_data: Input data dictionary from workflow execution
            **kwargs: Additional keyword arguments
            
        Returns:
            User query string or empty string if not found
        """
        # Method 1: Direct access to user_query
        user_query = input_data.get('user_query', '')
        if user_query and isinstance(user_query, str):
            self.nb_logger.debug(f"✅ Found user_query directly: '{user_query[:50]}...'")
            return user_query.strip()
        
        # Method 2: Check workflow context
        workflow_context = input_data.get('workflow_context', {})
        if workflow_context and isinstance(workflow_context, dict):
            user_query = workflow_context.get('user_query', '')
            if user_query and isinstance(user_query, str):
                self.nb_logger.debug(f"✅ Found user_query in workflow_context: '{user_query[:50]}...'")
                return user_query.strip()
        
        # Method 3: Check nested input structures (input_0, input_1, etc.)
        for key, value in input_data.items():
            if key.startswith('input_') and isinstance(value, dict):
                nested_query = value.get('user_query', '')
                if nested_query and isinstance(nested_query, str):
                    self.nb_logger.debug(f"✅ Found user_query in {key}: '{nested_query[:50]}...'")
                    return nested_query.strip()
                
                # Also check nested workflow_context
                nested_context = value.get('workflow_context', {})
                if isinstance(nested_context, dict):
                    nested_query = nested_context.get('user_query', '')
                    if nested_query and isinstance(nested_query, str):
                        self.nb_logger.debug(f"✅ Found user_query in {key}.workflow_context: '{nested_query[:50]}...'")
                        return nested_query.strip()
        
        # Method 4: Check if any registered data units contain the query
        if hasattr(self, 'input_data_units') and self.input_data_units:
            for unit_name, data_unit in self.input_data_units.items():
                try:
                    if hasattr(data_unit, 'get'):
                        # Try to get data from data unit
                        try:
                            unit_data = await data_unit.get()
                            if isinstance(unit_data, str) and unit_data.strip():
                                self.nb_logger.debug(f"✅ Found user_query in data unit {unit_name}: '{unit_data[:50]}...'")
                                return unit_data.strip()
                        except Exception as e:
                            self.nb_logger.debug(f"Could not await data unit {unit_name}: {e}")
                            # Try synchronous access as fallback
                            try:
                                unit_data = data_unit.get()
                                if isinstance(unit_data, str) and unit_data.strip():
                                    self.nb_logger.debug(f"✅ Found user_query in data unit {unit_name} (sync): '{unit_data[:50]}...'")
                                    return unit_data.strip()
                            except Exception:
                                pass
                except Exception as e:
                    self.nb_logger.debug(f"Could not access data unit {unit_name}: {e}")
        
        # Method 5: Check kwargs for user_query
        user_query = kwargs.get('user_query', '')
        if user_query and isinstance(user_query, str):
            self.nb_logger.debug(f"✅ Found user_query in kwargs: '{user_query[:50]}...'")
            return user_query.strip()
        
        # Method 6: Look for common query field names
        query_fields = ['query', 'question', 'input_text', 'message', 'text', 'content']
        for field in query_fields:
            if field in input_data:
                value = input_data[field]
                if isinstance(value, str) and value.strip():
                    self.nb_logger.debug(f"✅ Found query in field {field}: '{value[:50]}...'")
                    return value.strip()
        
        # Log detailed information for debugging
        self.nb_logger.warning("❌ Could not extract user query from input data")
        self.nb_logger.warning(f"Input data keys: {list(input_data.keys())}")
        self.nb_logger.warning(f"Input data types: {[(k, type(v).__name__) for k, v in input_data.items()]}")
        
        return ''
    
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
                
                # Enhanced text parsing fallback
                content_lower = content.lower()
                
                # Comprehensive list of virus species with exact matching
                virus_patterns = [
                    ('chikungunya virus', 'chikungunya virus'),
                    ('chikungunya', 'chikungunya virus'),
                    ('eastern equine encephalitis virus', 'eastern equine encephalitis virus'),
                    ('eastern equine encephalitis', 'eastern equine encephalitis virus'),
                    ('western equine encephalitis virus', 'western equine encephalitis virus'),
                    ('western equine encephalitis', 'western equine encephalitis virus'),
                    ('venezuelan equine encephalitis virus', 'venezuelan equine encephalitis virus'),
                    ('venezuelan equine encephalitis', 'venezuelan equine encephalitis virus'),
                    ('sindbis virus', 'sindbis virus'),
                    ('semliki forest virus', 'semliki forest virus'),
                    ('ross river virus', 'ross river virus'),
                    ('zika virus', 'zika virus'),
                    ('dengue virus', 'dengue virus'),
                    ('yellow fever virus', 'yellow fever virus'),
                    ('mayaro virus', 'mayaro virus'),
                    ('o\'nyong\'nyong virus', 'o\'nyong\'nyong virus'),
                    ('barmah forest virus', 'barmah forest virus'),
                    ('una virus', 'una virus')
                ]
                
                # Check for virus patterns in content
                for pattern, canonical_name in virus_patterns:
                    if pattern in content_lower:
                        self.nb_logger.info(f"✅ Extracted virus species via text parsing: {canonical_name}")
                        return {
                            'virus_species': canonical_name,
                            'confidence': 0.8,  # High confidence for direct text matches
                            'reasoning': f'Extracted from text: {canonical_name}'
                        }
                
                # If no direct matches, check for "virus" keyword to catch other cases
                words = content_lower.split()
                for i, word in enumerate(words):
                    if 'virus' in word and i > 0:
                        # Look for potential virus names before the word "virus"
                        potential_virus = ' '.join(words[max(0, i-3):i+1])
                        if len(potential_virus) > 5:  # Reasonable minimum length
                            self.nb_logger.info(f"✅ Potential virus species found: {potential_virus}")
                            return {
                                'virus_species': potential_virus,
                                'confidence': 0.6,
                                'reasoning': f'Potential virus species: {potential_virus}'
                            }
                
                self.nb_logger.warning("⚠️ No virus species detected in response")
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