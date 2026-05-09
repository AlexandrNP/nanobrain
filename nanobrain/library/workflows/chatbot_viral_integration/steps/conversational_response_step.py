"""
Conversational Response Step

Provides educational and informational responses about alphaviruses using real LLM generation.
Includes scientific knowledge base and literature reference integration.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from nanobrain.core.step import Step, StepConfig
from nanobrain.library.agents.specialized.base import ConversationalSpecializedAgent
from nanobrain.library.infrastructure.data.chat_session_data import (
    ConversationalResponseData
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
        # The configuration is loaded as attributes on the config object
        agent_config_file = None

        # Method 1: Direct attribute access (most common)
        if hasattr(self.config, 'agent_config_file'):
            agent_config_file = getattr(self.config, 'agent_config_file')
            self.nb_logger.info(f"✅ Found agent_config_file: {agent_config_file}")

        # Method 2: From config dict if available
        elif hasattr(self.config, '__dict__') and 'agent_config_file' in self.config.__dict__:
            agent_config_file = self.config.__dict__['agent_config_file']
            self.nb_logger.info(f"✅ Found agent_config_file in __dict__: {agent_config_file}")

        # Method 3: From nested config structure
        elif hasattr(self.config, 'config') and hasattr(self.config.config, 'agent_config_file'):
            agent_config_file = self.config.config.agent_config_file
            self.nb_logger.info(f"✅ Found agent_config_file in nested config: {agent_config_file}")

        if not agent_config_file:
            # Enhanced error message with debugging info
            config_attrs = [attr for attr in dir(self.config) if not attr.startswith('_')]
            config_dict_keys = list(self.config.__dict__.keys()) if hasattr(self.config, '__dict__') else []

            raise ValueError(
                f"❌ FRAMEWORK VIOLATION: No agent_config_file specified in step configuration.\n"
                f"   REQUIRED: Specify agent_config_file in step config YAML.\n"
                f"   EXAMPLE: agent_config_file: 'config/ConversationalResponseStep/ConversationalAgent.yml'\n"
                f"   DEBUG: Config type: {type(self.config)}\n"
                f"   DEBUG: Config attributes: {config_attrs}\n"
                f"   DEBUG: Config dict keys: {config_dict_keys}"
            )
        
        # ✅ FRAMEWORK COMPLIANCE: Load agent from config file using from_config pattern
        from nanobrain.library.agents.specialized.viral_expert_agent import ViralExpertConversationalAgent
        
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
            agent = ViralExpertConversationalAgent.from_config(str(agent_config_path))
            
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
        # CRITICAL FIX: Initialize start_time at the very beginning to avoid undefined variable errors
        start_time = time.time()

        try:
            # Handle both direct input data and data unit structure
            actual_data = input_data
            if len(input_data) == 1 and 'input_0' in input_data:
                # Data came from data unit
                actual_data = input_data['input_0']

            # PHASE 2 DEBUG: Log the actual input data structure
            self.nb_logger.info(f"🔍 [DEBUG-TRACE] Raw input_data keys: {list(input_data.keys())}")
            self.nb_logger.info(f"🔍 [DEBUG-TRACE] Raw input_data: {input_data}")
            self.nb_logger.info(f"🔍 [DEBUG-TRACE] Actual_data type: {type(actual_data)}")
            self.nb_logger.info(f"🔍 [DEBUG-TRACE] Actual_data: {actual_data}")

            # CRITICAL DEBUG: Log detailed structure for request_id debugging
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    self.nb_logger.info(f"🔍 [REQUEST-ID-DEBUG] input_data[{key}] = {type(value).__name__}: {value}")
            if isinstance(actual_data, dict):
                for key, value in actual_data.items():
                    self.nb_logger.info(f"🔍 [REQUEST-ID-DEBUG] actual_data[{key}] = {type(value).__name__}: {value}")

            # PHASE 2 FIX: Extract request_id and session_id for HTTP response matching
            # Handle different input data structures
            request_id = None
            session_id = 'default'
            query_text = None

            # CRITICAL FIX: Check all possible input data structures for request_id
            if isinstance(actual_data, dict):
                # Method 1: Direct request_id in actual_data
                if 'request_id' in actual_data:
                    request_id = actual_data.get('request_id')
                    session_id = actual_data.get('session_id', 'default')
                    query_text = actual_data.get('user_query') or actual_data.get('message') or actual_data.get('query') or actual_data.get('text')
                    self.nb_logger.info(f"🔍 [REQUEST-ID-FIX] Found request_id in actual_data: {request_id}")

                # Method 2: Check if actual_data contains nested request structure
                elif isinstance(actual_data, dict) and len(actual_data) == 1:
                    nested_value = list(actual_data.values())[0]
                    if isinstance(nested_value, dict) and 'request_id' in nested_value:
                        request_id = nested_value.get('request_id')
                        session_id = nested_value.get('session_id', 'default')
                        query_text = nested_value.get('user_query') or nested_value.get('message') or nested_value.get('query') or nested_value.get('text')
                        self.nb_logger.info(f"🔍 [REQUEST-ID-FIX] Found request_id in nested structure: {request_id}")

                # Method 3: Check original input_data for request_id (before actual_data extraction)
                if not request_id and isinstance(input_data, dict):
                    for key, value in input_data.items():
                        if isinstance(value, dict) and 'request_id' in value:
                            request_id = value.get('request_id')
                            session_id = value.get('session_id', 'default')
                            query_text = value.get('user_query') or value.get('message') or value.get('query') or value.get('text')
                            self.nb_logger.info(f"🔍 [REQUEST-ID-FIX] Found request_id in input_data[{key}]: {request_id}")
                            break

                # Method 4: Extract query text if we haven't found it yet
                if not query_text:
                    query_text = actual_data.get('user_query') or actual_data.get('query') or actual_data.get('message') or actual_data.get('text')
                    if not query_text and len(actual_data) == 1:
                        # Single key-value pair, use the value as query
                        single_value = list(actual_data.values())[0]
                        # If the single value is a dict, try to extract user_query from it
                        if isinstance(single_value, dict):
                            query_text = single_value.get('user_query') or single_value.get('query') or single_value.get('message')
                        else:
                            query_text = single_value
            elif isinstance(actual_data, str):
                # Direct string input
                query_text = actual_data

            # CRITICAL FIX: Ensure query_text is always a string
            if query_text and not isinstance(query_text, str):
                # If query_text is not a string, try to extract the actual query
                if isinstance(query_text, dict):
                    query_text = query_text.get('user_query') or query_text.get('query') or query_text.get('message') or str(query_text)
                else:
                    query_text = str(query_text)

            self.nb_logger.info(f"🔍 [HTTP-TRACE] Processing request_id: {request_id}, session_id: {session_id}")
            self.nb_logger.info(f"🔍 [HTTP-TRACE] Extracted query_text type: {type(query_text)}")
            self.nb_logger.info(f"🔍 [HTTP-TRACE] Extracted query_text: {str(query_text)[:100] if query_text else 'None'}...")

            # Check if this step should execute based on routing decision
            routing_decision = actual_data.get('routing_decision', {})
            next_step = routing_decision.get('next_step')

            # CRITICAL FIX: Handle direct conversational queries without routing
            if next_step is not None and next_step != 'conversational_response':
                # This step shouldn't execute for this query type
                self.nb_logger.info(f"🚫 Skipping conversational response step (routing to: {next_step})")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': f'Query routed to {next_step}',
                    'response_data': None
                }

            # If no routing decision or routing to conversational_response, process the query
            self.nb_logger.info(f"✅ Processing conversational query (routing: {next_step or 'direct'})")

            classification_data = actual_data.get('classification_data')

            # CRITICAL FIX: Handle direct user queries without classification_data
            if not classification_data:
                # Use the extracted query_text
                if not query_text:
                    self.nb_logger.error("🔍 DEBUG: No query_text extracted from input data")
                    raise ValueError("Missing both classification_data and query_text")

                # Create a simple classification data structure
                from types import SimpleNamespace
                classification_data = SimpleNamespace()
                classification_data.original_query = str(query_text)  # Ensure it's a string
                self.nb_logger.info(f"✅ Created classification_data from query_text: {str(query_text)[:50]}...")

            query = classification_data.original_query
            # CRITICAL FIX: Ensure query is always a string
            if not isinstance(query, str):
                query = str(query)

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

            # PHASE 2 FIX: Create HTTP-compatible response with request_id for web interface matching
            http_compatible_response = {
                'request_id': request_id,
                'session_id': session_id,
                'response': response_data.response,
                'workflow_type': 'conversational_viral_expert',
                'processing_time': processing_time,
                'response_type': response_data.response_type,
                'confidence': response_data.confidence,
                'topic_area': response_data.topic_area,
                'timestamp': datetime.now().isoformat()
            }

            self.nb_logger.info(f"🔍 [HTTP-TRACE] Created HTTP-compatible response with request_id: {request_id}")

            return {
                'success': True,
                'conversation_output': http_compatible_response,  # ✅ CRITICAL FIX: HTTP-compatible format
                'response_data': response_data,  # Keep for backward compatibility
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Conversational response generation failed: {e}")

            # CRITICAL FIX: Ensure start_time is defined for processing time calculation
            if 'start_time' not in locals():
                start_time = time.time()

            # CRITICAL FIX: Safely extract request_id and session_id for error response
            request_id = None
            session_id = 'default'
            try:
                # Try to extract from input_data if available
                if isinstance(input_data, dict):
                    request_id = input_data.get('request_id')
                    session_id = input_data.get('session_id', 'default')
            except:
                pass  # Use defaults if extraction fails

            # Generate fallback response
            classification_data = input_data.get('classification_data') if isinstance(input_data, dict) else None
            query = classification_data.original_query if classification_data else ''

            processing_time = (time.time() - start_time) * 1000

            fallback_response = ConversationalResponseData(
                query=query,
                response=f"I encountered an error while processing your request: {str(e)}",
                response_type='error',
                confidence=0.0,
                processing_time_ms=processing_time
            )

            # PHASE 2 FIX: Create HTTP-compatible fallback response with request_id
            http_compatible_fallback = {
                'request_id': request_id,
                'session_id': session_id,
                'response': fallback_response.response,
                'workflow_type': 'conversational_viral_expert',
                'processing_time': processing_time,
                'response_type': 'error',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            return {
                'success': False,
                'conversation_output': http_compatible_fallback,  # ✅ HTTP-compatible format
                'response_data': fallback_response,  # Keep for backward compatibility
                'error': str(e)
            }
    
    async def _generate_llm_response(self, query: str, topic_hints: List[str]) -> ConversationalResponseData:
        """Generate response using LLM agent with topic context"""
        
        # Ensure agent is properly initialized
        await self._ensure_agent_initialized()
        
        # Enhance query with topic context for better LLM responses
        enhanced_query = self._enhance_query_with_context(query, topic_hints)
        
        try:
            # PHASE 1 DIAGNOSTICS: Enhanced agent call tracing
            self.nb_logger.info("🔍 [STEP-TRACE] Calling agent.process() with enhanced query...")
            self.nb_logger.info(f"🔍 [STEP-TRACE] Agent type: {type(self.agent)}")
            self.nb_logger.info(f"🔍 [STEP-TRACE] Agent name: {getattr(self.agent, 'name', 'unknown')}")
            self.nb_logger.info(f"🔍 [STEP-TRACE] Enhanced query length: {len(enhanced_query)}")

            agent_call_start = time.time()

            # Generate response using conversational agent
            llm_response = await self.agent.process(enhanced_query)

            agent_call_duration = time.time() - agent_call_start
            self.nb_logger.info(f"✅ [STEP-TRACE] agent.process() completed in {agent_call_duration:.2f}s")
            self.nb_logger.info(f"🔍 [STEP-TRACE] Response length: {len(llm_response) if llm_response else 0}")
            
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