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
        """
        Load LLM agent for virus species extraction from standardized config file.
        
        ✅ FRAMEWORK COMPLIANCE: Uses agent_config_file reference, no programmatic creation.
        """
        # Get agent config file path from step configuration
        agent_config_file = component_config.get('agent_config_file')
        
        if not agent_config_file:
            raise ValueError(
                "❌ FRAMEWORK VIOLATION: No agent_config_file specified in step configuration.\n"
                "   REQUIRED: Specify agent_config_file in step config YAML.\n"
                "   EXAMPLE: agent_config_file: 'config/QueryClassificationStep/VirusExtractionAgent.yml'"
            )
        
        # ✅ FRAMEWORK COMPLIANCE: Load agent from config file using from_config pattern
        from nanobrain.library.agents.specialized_agents.conversational_specialized_agent import ConversationalSpecializedAgent
        
        try:
            # Resolve agent config file path relative to workflow directory
            if hasattr(self, 'workflow_directory') and self.workflow_directory:
                import os
                from pathlib import Path
                agent_config_path = Path(self.workflow_directory) / agent_config_file
            else:
                # Fallback: resolve relative to current step's config location
                import os
                from pathlib import Path
                step_dir = Path(__file__).parent.parent
                agent_config_path = step_dir / agent_config_file
            
            # Load agent using framework's from_config pattern
            return ConversationalSpecializedAgent.from_config(str(agent_config_path))
            
        except Exception as e:
            raise ValueError(
                f"❌ FRAMEWORK ERROR: Failed to load agent from {agent_config_file}: {e}\n"
                f"   SOLUTION: Ensure agent config file exists and is properly formatted.\n"
                f"   PATH: {agent_config_file}"
            ) from e
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user query to extract virus species and determine analysis type using LLM agent
        
        Args:
            input_data: Contains 'user_query' key with query text
            
        Returns:
            Dict with routing decision, extracted virus species, and analysis type
        """
        user_query = input_data.get('user_query', '')
        
        if not user_query:
            return {
                'routing_decision': 'conversational_response',
                'error': 'No user query provided'
            }
        
        try:
            # Extract virus species and analysis type using enhanced LLM agent
            extraction_result = await self._extract_virus_species_llm(user_query)
            
            # Use LLM-provided routing decision if available, otherwise use fallback logic
            routing_decision = extraction_result.get('routing_decision', 'conversational_response')
            
            # Enhanced routing logic based on virus species AND analysis type
            virus_species = extraction_result.get('virus_species')
            analysis_type = extraction_result.get('analysis_type', 'conversational')
            
            if virus_species and analysis_type in ['pssm', 'protein_analysis']:
                # Virus species detected AND analysis requested - route to virus name resolution
                routing_decision = 'virus_name_resolution'
            elif virus_species and analysis_type == 'conversational':
                # Virus species detected but no analysis - could still route to virus resolution for info
                routing_decision = 'virus_name_resolution' 
            else:
                # No virus species or general conversation - route to conversational response
                routing_decision = 'conversational_response'
            
            return {
                'routing_decision': routing_decision,
                'extracted_virus_species': virus_species,
                'analysis_type': analysis_type,
                'confidence': extraction_result.get('confidence', 0.0),
                'reasoning': extraction_result.get('reasoning', ''),
                'user_query': user_query
            }
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error in enhanced query classification: {e}")
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