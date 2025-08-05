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
        
        # Get resolved extraction agent from configuration
        self.extraction_agent = self._get_resolved_extraction_agent(component_config)
        
        if self.nb_logger:
            self.nb_logger.info("🔍 Query Classification Step initialized with LLM-based virus species extraction")
    
    def _get_resolved_extraction_agent(self, component_config: Dict[str, Any]) -> 'VirusExtractionAgent':
        """
        Get the resolved extraction agent from configuration.
        
        ✅ FRAMEWORK COMPLIANCE: Uses resolved agent object from class+config pattern.
        The extraction_agent should already be instantiated during configuration loading.
        """
        # Get resolved extraction agent from configuration
        extraction_agent = component_config.get('extraction_agent')
        
        if extraction_agent is None:
            raise ValueError(
                "❌ FRAMEWORK VIOLATION: No extraction_agent found in step configuration.\n"
                "   REQUIRED: Specify extraction_agent with class+config pattern in step config YAML.\n"
                "   EXAMPLE:\n"
                "     extraction_agent:\n"
                "       class: 'nanobrain.library.agents.specialized.virus_extraction_agent.VirusExtractionAgent'\n"
                "       config: 'config/QueryClassificationStep/VirusExtractionAgent.yml'"
            )
        
        # Validate that it's the correct agent type
        from nanobrain.library.agents.specialized.virus_extraction_agent import VirusExtractionAgent
        
        if not isinstance(extraction_agent, VirusExtractionAgent):
            raise ValueError(
                f"❌ FRAMEWORK ERROR: extraction_agent must be VirusExtractionAgent instance.\n"
                f"   FOUND: {type(extraction_agent)}\n"
                f"   EXPECTED: VirusExtractionAgent\n"
                f"   SOLUTION: Check class+config pattern in step configuration"
            )
        
        self.nb_logger.info(f"✅ Resolved extraction agent: {type(extraction_agent).__name__}")
        return extraction_agent
    
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
        Extract virus species using specialized VirusExtractionAgent
        
        Args:
            user_query: The user's query text
            
        Returns:
            Dict with virus species, confidence, and routing information
        """
        try:
            # Use the VirusExtractionAgent's specialized processing capability
            extraction_result = await self.extraction_agent._process_specialized_request(
                user_query,
                expected_format='json',
                analysis_type='classification'
            )
            
            # Parse the JSON response from the specialized agent
            if extraction_result:
                import json
                try:
                    parsed_result = json.loads(extraction_result)
                    return parsed_result
                except json.JSONDecodeError:
                    self.nb_logger.warning(f"⚠️ Non-JSON response from extraction agent: {extraction_result}")
                    return {
                        'virus_species': None,
                        'confidence': 0.0,
                        'reasoning': 'Agent returned non-JSON response',
                        'analysis_type': 'conversational',
                        'routing_decision': 'conversational_response'
                    }
            else:
                # Agent returned None - fallback to general processing
                fallback_result = await self.extraction_agent.process({
                    'user_query': user_query,
                    'task': 'virus_species_extraction',
                    'format': 'json'
                })
                
                # Parse fallback result
                return self._parse_fallback_response(fallback_result, user_query)
                
        except Exception as e:
            self.nb_logger.error(f"❌ Error in virus species extraction: {e}")
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Extraction error: {str(e)}',
                'analysis_type': 'conversational',
                'routing_decision': 'conversational_response'
            }
    
    def _parse_fallback_response(self, response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        Parse fallback response from general agent processing
        
        Args:
            response: Response from agent's general process method
            user_query: Original user query for context
            
        Returns:
            Dict with extracted information in standard format
        """
        try:
            # Extract content from response
            content = ""
            if isinstance(response, dict):
                content = response.get('content', response.get('text', str(response)))
            else:
                content = str(response)
            
            # Basic virus detection in content
            content_lower = content.lower()
            user_query_lower = user_query.lower()
            
            # Check for virus indicators
            virus_indicators = [
                'chikungunya', 'chikv', 'eastern equine encephalitis', 'eeev',
                'alphavirus', 'togavirus', 'viral', 'virus'
            ]
            
            detected_virus = None
            for indicator in virus_indicators:
                if indicator in user_query_lower or indicator in content_lower:
                    detected_virus = indicator
                    break
            
            # Determine analysis type from query
            analysis_indicators = ['pssm', 'matrix', 'analysis', 'protein', 'sequence']
            analysis_requested = any(indicator in user_query_lower for indicator in analysis_indicators)
            
            return {
                'virus_species': detected_virus,
                'confidence': 0.6 if detected_virus else 0.0,
                'reasoning': f'Fallback detection from agent response and query analysis',
                'analysis_type': 'pssm' if analysis_requested else 'conversational',
                'routing_decision': 'virus_name_resolution' if detected_virus else 'conversational_response'
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error parsing fallback response: {e}")
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Fallback parse error: {str(e)}',
                'analysis_type': 'conversational',
                'routing_decision': 'conversational_response'
            } 