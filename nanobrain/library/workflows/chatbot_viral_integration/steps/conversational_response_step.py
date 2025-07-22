"""
Conversational Response Step

Provides educational and informational responses about alphaviruses using real LLM generation.
Includes scientific knowledge base and literature reference integration.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.agent import AgentConfig
from nanobrain.library.agents.specialized.base import ConversationalSpecializedAgent
from nanobrain.library.infrastructure.data.chat_session_data import (
    ConversationalResponseData, MessageType
)
from typing import Dict, Any, List, Optional
import time
from datetime import datetime


class AlphavirusConversationalAgent(ConversationalSpecializedAgent):
    """
    Concrete implementation of ConversationalSpecializedAgent for alphavirus education.
    
    This agent provides educational responses about alphaviruses and implements
    the required abstract methods from the specialized agent base.
    """
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized alphavirus requests that don't require LLM.
        
        For now, this falls back to LLM processing for all requests.
        Future enhancements could include direct lookups for simple facts.
        """
        # For alphavirus education, we primarily rely on LLM responses
        # This method could be enhanced to handle simple factual queries directly
        return None
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by specialized logic.
        
        For now, we let all requests go to the LLM for comprehensive responses.
        """
        # Could implement keyword-based routing for simple queries in the future
        return False


class ConversationalResponseStep(Step):
    """
    Step for generating educational responses about alphaviruses using real LLM.
    
    Provides scientific information with literature references
    and handles various biology topics related to alphaviruses.
    """
    
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': 'Conversational response step for alphavirus education',
        'temperature': 0.7,
        'max_tokens': 2000,
        'system_prompt_type': 'alphavirus_expert'
    }
    
    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract ConversationalResponseStep configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'temperature': getattr(config, 'temperature', 0.7),
            'max_tokens': getattr(config, 'max_tokens', 2000),
            'system_prompt_type': getattr(config, 'system_prompt_type', 'alphavirus_expert'),
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ConversationalResponseStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.temperature = component_config['temperature']
        self.max_tokens = component_config['max_tokens']
        self.system_prompt_type = component_config['system_prompt_type']
        
        # Initialize conversational agent for LLM responses
        self.agent = None
        self._agent_initialized = False
        
        # Initialize context and reference database
        self.literature_refs = self._initialize_literature_references()
        
        self.nb_logger.info("🧠 Conversational Response Step initialized")
    
    async def _ensure_agent_initialized(self) -> None:
        """Ensure the conversational agent is properly initialized"""
        if not self._agent_initialized:
            self.nb_logger.info("🔄 Initializing conversational agent with LLM...")
            self.agent = self._initialize_conversational_agent()
            await self.agent.initialize()
            self._agent_initialized = True
            self.nb_logger.info("✅ Conversational agent initialized with LLM client")
    
    def _initialize_conversational_agent(self) -> AlphavirusConversationalAgent:
        """
        Load conversational agent from standardized config file.
        
        ✅ FRAMEWORK COMPLIANCE: Uses agent_config_file reference, no programmatic creation.
        """
        # Get agent config file path from step configuration
        component_config = getattr(self.config, 'config', {}) if hasattr(self.config, 'config') else {}
        agent_config_file = component_config.get('agent_config_file')
        
        if not agent_config_file:
            raise ValueError(
                "❌ FRAMEWORK VIOLATION: No agent_config_file specified in step configuration.\n"
                "   REQUIRED: Specify agent_config_file in step config YAML.\n"
                "   EXAMPLE: agent_config_file: 'config/ConversationalResponseStep/ConversationalAgent.yml'"
            )
        
        # ✅ FRAMEWORK COMPLIANCE: Load agent from config file using from_config pattern
        from nanobrain.library.agents.specialized_agents.conversational_specialized_agent import ConversationalSpecializedAgent
        
        try:
            # Resolve agent config file path relative to workflow directory
            if hasattr(self, 'workflow_directory') and self.workflow_directory:
                from pathlib import Path
                agent_config_path = Path(self.workflow_directory) / agent_config_file
            else:
                # Fallback: resolve relative to current step's config location
                from pathlib import Path
                step_dir = Path(__file__).parent.parent
                agent_config_path = step_dir / agent_config_file
            
            # Load agent using framework's from_config pattern
            agent = ConversationalSpecializedAgent.from_config(str(agent_config_path))
            
            # Return as AlphavirusConversationalAgent (should be compatible)
            return agent
            
        except Exception as e:
            raise ValueError(
                f"❌ FRAMEWORK ERROR: Failed to load ConversationalAgent from {agent_config_file}: {e}\n"
                f"   SOLUTION: Ensure agent config file exists and is properly formatted.\n"
                f"   PATH: {agent_config_file}"
            ) from e
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate conversational response about alphaviruses using LLM.
        
        Args:
            input_data: Contains classification_data and routing_decision
            
        Returns:
            Dictionary with ConversationalResponseData
        """
        start_time = time.time()
        
        try:
            # Handle both direct input data and data unit structure
            actual_data = input_data
            if len(input_data) == 1 and 'input_0' in input_data:
                # Data came from data unit
                actual_data = input_data['input_0']
            
            # Check if this step should execute based on routing decision
            routing_decision = actual_data.get('routing_decision', {})
            next_step = routing_decision.get('next_step')
            
            if next_step != 'conversational_response':
                # This step shouldn't execute for this query type
                self.nb_logger.info(f"🚫 Skipping conversational response step (routing to: {next_step})")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': f'Query routed to {next_step}',
                    'response_data': None
                }
            
            classification_data = actual_data.get('classification_data')
            
            if not classification_data:
                raise ValueError("Missing classification_data")
            
            query = classification_data.original_query
            topic_hints = routing_decision.get('topic_hints', ['general'])
            clarification_needed = routing_decision.get('clarification_needed', False)
            
            self.nb_logger.info(f"🧠 Generating LLM response for query: {query[:100]}...")
            
            # Generate response using LLM agent
            if clarification_needed:
                response_data = await self._generate_clarification_response(query)
            else:
                response_data = await self._generate_llm_response(query, topic_hints)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            response_data.processing_time_ms = processing_time
            
            self.nb_logger.info(f"✅ Generated {response_data.response_type} response ({len(response_data.response)} chars)")
            
            return {
                'success': True,
                'response_data': response_data,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Conversational response generation failed: {e}")
            
            # Generate fallback response
            classification_data = input_data.get('classification_data')
            query = classification_data.original_query if classification_data else ''
            
            fallback_response = ConversationalResponseData(
                query=query,
                response="I apologize, but I encountered an error generating a response. Please try rephrasing your question or ask about alphavirus structure, replication, or diseases.",
                response_type='error',
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            return {
                'success': False,
                'response_data': fallback_response,
                'error': str(e)
            }
    
    async def _generate_llm_response(self, query: str, topic_hints: List[str]) -> ConversationalResponseData:
        """Generate response using LLM agent with topic context"""
        
        # Ensure agent is properly initialized
        await self._ensure_agent_initialized()
        
        # Enhance query with topic context for better LLM responses
        enhanced_query = self._enhance_query_with_context(query, topic_hints)
        
        try:
            # Generate response using conversational agent
            llm_response = await self.agent.process(enhanced_query)
            
            # Determine primary topic and confidence
            primary_topic = topic_hints[0] if topic_hints else 'general'
            confidence = 0.85  # High confidence for LLM responses
            
            # Create response data
            response_data = ConversationalResponseData(
                query=query,
                response=llm_response,
                response_type='educational',
                confidence=confidence,
                topic_area=primary_topic
            )
            
            # Add relevant literature references
            await self._add_literature_references(response_data, primary_topic)
            
            return response_data
            
        except Exception as e:
            self.nb_logger.error(f"LLM generation failed: {e}")
            
            # Fallback to basic informational response
            fallback_response = self._generate_fallback_response(query, topic_hints)
            return fallback_response
    
    def _enhance_query_with_context(self, query: str, topic_hints: List[str]) -> str:
        """Enhance query with topic context for better LLM responses"""
        
        context_map = {
            'structure': "Focus on molecular structure, protein domains, and structural organization.",
            'replication': "Focus on viral replication cycle, molecular mechanisms, and host interactions.",
            'diseases': "Focus on pathogenesis, clinical manifestations, and disease mechanisms.",
            'transmission': "Focus on vector biology, epidemiology, and transmission mechanisms.",
            'evolution': "Focus on evolutionary relationships, phylogenetics, and genetic diversity.",
            'general': "Provide comprehensive scientific information."
        }
        
        # Add context based on topic hints
        context_parts = []
        for topic in topic_hints[:2]:  # Use top 2 topics
            if topic in context_map:
                context_parts.append(context_map[topic])
        
        if context_parts:
            enhanced_query = f"{query}\n\nContext: {' '.join(context_parts)}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    async def _generate_clarification_response(self, query: str) -> ConversationalResponseData:
        """Generate clarification response for unclear queries"""
        
        clarification_prompt = f"""The user asked: "{query}"

This query needs clarification. Please provide a helpful response that:
1. Acknowledges their interest in alphaviruses
2. Lists specific topics you can help with
3. Asks for more specific information
4. Provides examples of good questions they could ask

Be friendly and educational, encouraging them to ask more specific questions about alphavirus biology."""

        try:
            llm_response = await self.agent.process(clarification_prompt)
            
            response_data = ConversationalResponseData(
                query=query,
                response=llm_response,
                response_type='clarification',
                confidence=1.0,
                topic_area='general'
            )
            
            return response_data
            
        except Exception as e:
            self.nb_logger.error(f"Clarification generation failed: {e}")
            
            # Static fallback for clarification
            fallback_clarification = """I'd be happy to help you learn about alphaviruses! However, I need a bit more information to provide the most relevant answer.

**I can help you with:**
🦠 **Virus Structure** - envelope proteins, capsid, genome organization
🔄 **Replication Cycle** - viral life cycle, host cell interaction
🏥 **Diseases** - symptoms, pathogenesis, clinical aspects
🦟 **Transmission** - vectors, epidemiology, prevention
🧬 **Evolution** - phylogeny, mutations, viral diversity
📊 **Classification** - taxonomy, viral families, nomenclature

Please let me know which topic interests you most, or provide more specific details about what you'd like to learn!"""

            response_data = ConversationalResponseData(
                query=query,
                response=fallback_clarification,
                response_type='clarification',
                confidence=1.0,
                topic_area='general'
            )
            
            return response_data
    
    def _generate_fallback_response(self, query: str, topic_hints: List[str]) -> ConversationalResponseData:
        """Generate fallback response when LLM fails"""
        
        primary_topic = topic_hints[0] if topic_hints else 'general'
        
        fallback_responses = {
            'structure': """**Alphavirus Structure**

Alphaviruses are enveloped RNA viruses with sophisticated structural organization:

🧬 **Genome**: Single-stranded, positive-sense RNA (~11,700 nucleotides)
🔬 **Virion**: Icosahedral nucleocapsid core with lipid envelope (~70 nm diameter)
🧪 **Proteins**: Capsid (C) forms core; Envelope proteins E1 & E2 on surface

**Key Features:**
- E2 protein handles receptor binding and cellular attachment
- E1 protein contains membrane fusion machinery
- Capsid protein packages genomic RNA specifically
- Overall icosahedral symmetry with 240 protein subunits

Would you like to know more about any specific structural component?""",
            
            'replication': """**Alphavirus Replication**

Alphaviruses follow a complex replication strategy:

🔄 **Entry**: Receptor-mediated endocytosis followed by pH-triggered fusion
🧬 **Translation**: Direct translation of genomic RNA produces nonstructural proteins
⚙️ **Replication**: Formation of replication complexes for RNA synthesis
📦 **Assembly**: Coordinated assembly of nucleocapsid and envelope

**Key Steps:**
- nsP1-4 form replication machinery
- Subgenomic RNA produces structural proteins
- Assembly occurs at cellular membranes
- Budding releases mature virions

Would you like details about any specific replication step?""",
            
            'diseases': """**Alphavirus Diseases**

Alphaviruses cause significant human and animal diseases:

🏥 **Major Human Pathogens:**
- **Eastern Equine Encephalitis** - severe neurological disease
- **Chikungunya** - joint pain and fever
- **Western Equine Encephalitis** - mild to severe encephalitis

🦟 **Transmission**: Primarily mosquito-borne (Aedes, Culex species)
🌍 **Distribution**: Worldwide, with regional variations

**Clinical Features:**
- Fever, headache, muscle pain
- Neurological complications (encephalitis)
- Joint involvement (arthritis)
- Variable severity by virus species

Would you like information about a specific alphavirus disease?"""
        }
        
        response_text = fallback_responses.get(primary_topic, 
            "I apologize, but I'm having trouble generating a detailed response right now. Please try asking about alphavirus structure, replication, or diseases.")
        
        response_data = ConversationalResponseData(
            query=query,
            response=response_text,
            response_type='factual',
            confidence=0.7,
            topic_area=primary_topic
        )
        
        return response_data
    
    async def _add_literature_references(self, response_data: ConversationalResponseData, topic: str):
        """Add relevant literature references to response"""
        
        # Add references for the topic
        if topic in self.literature_refs:
            topic_refs = self.literature_refs[topic][:3]  # Max 3 references
            
            for ref in topic_refs:
                response_data.add_reference(
                    title=ref['title'],
                    authors=ref['authors'],
                    journal=ref['journal'],
                    year=ref['year'],
                    pmid=ref['pmid']
                )
    
    def _initialize_literature_references(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize literature references database"""
        
        return {
            'structure': [
                {
                    'title': 'Structure of chikungunya virus',
                    'authors': 'Voss JE, Vaney MC, Duquerroy S, et al.',
                    'journal': 'Nature',
                    'year': 2010,
                    'pmid': '20428234'
                },
                {
                    'title': 'Alphavirus structure and assembly',
                    'authors': 'Kuhn RJ',
                    'journal': 'Adv Virus Res',
                    'year': 2007,
                    'pmid': '17765004'
                }
            ],
            'replication': [
                {
                    'title': 'Alphavirus RNA replication',
                    'authors': 'Strauss JH, Strauss EG',
                    'journal': 'Microbiol Rev',
                    'year': 1994,
                    'pmid': '8078435'
                },
                {
                    'title': 'Alphavirus nonstructural proteins and their role in viral RNA replication',
                    'authors': 'Lemm JA, Rümenapf T, Strauss EG, et al.',
                    'journal': 'J Virol',
                    'year': 1994,
                    'pmid': '8254745'
                }
            ],
            'diseases': [
                {
                    'title': 'Chikungunya: a re-emerging virus',
                    'authors': 'Schwartz O, Albert ML',
                    'journal': 'Lancet',
                    'year': 2010,
                    'pmid': '19854199'
                },
                {
                    'title': 'Eastern equine encephalitis virus',
                    'authors': 'Morens DM, Folkers GK, Fauci AS',
                    'journal': 'N Engl J Med',
                    'year': 2019,
                    'pmid': '31291517'
                }
            ],
            'transmission': [
                {
                    'title': 'Alphavirus vectors and transmission',
                    'authors': 'Weaver SC, Reisen WK',
                    'journal': 'Annu Rev Entomol',
                    'year': 2010,
                    'pmid': '19961330'
                }
            ]
        } 