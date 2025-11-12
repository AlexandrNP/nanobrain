#!/usr/bin/env python3
"""
Response Synthesis Agent for RAG systems.
Intelligent information synthesis using multi-turn LLM capabilities.
"""

import json
import time
from typing import Dict, Any, List, Optional
from nanobrain.core.agent import ConversationalAgent
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class ResponseSynthesisAgent(ConversationalAgent):
    """
    Intelligent response synthesis agent for RAG systems.
    
    Leverages agentic LLM capabilities for intelligent information synthesis:
    - Analyzes relationships between retrieved document chunks
    - Identifies key themes and concepts
    - Resolves contradictions or inconsistencies
    - Creates coherent narrative flow
    - Implements multi-turn LLM strategy for comprehensive responses
    
    All prompts and templates are loaded from YAML configuration with zero hardcoding.
    """
    
    COMPONENT_TYPE = "response_synthesis_agent"
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize ResponseSynthesisAgent using framework pattern."""
        # Call parent initialization first
        super()._init_from_config(config, component_config, dependencies)

        # Store configuration
        self.config = config if hasattr(config, '__dict__') else component_config
        self.name = getattr(config, 'name', component_config.get('name', 'response_synthesis_agent'))
        self.initialized = False

        # Synthesis configuration
        self.max_response_length = getattr(config, 'max_response_length', component_config.get('max_response_length', 4000))
        self.max_chunks_per_turn = getattr(config, 'max_chunks_per_turn', component_config.get('max_chunks_per_turn', 5))
        self.enable_multi_turn = getattr(config, 'enable_multi_turn', component_config.get('enable_multi_turn', True))
        self.enable_contradiction_resolution = getattr(config, 'enable_contradiction_resolution', component_config.get('enable_contradiction_resolution', True))
        self.enable_theme_analysis = getattr(config, 'enable_theme_analysis', component_config.get('enable_theme_analysis', True))

        # Template configuration - all templates from YAML config
        self.template_variables = getattr(config, 'template_variables', component_config.get('template_variables', {}))

        # Initialize template system from configuration
        self.synthesis_template = self._load_template_from_config('synthesis_template')
        self.analysis_template = self._load_template_from_config('analysis_template')
        self.continuation_template = self._load_template_from_config('continuation_template')
        self.finalization_template = self._load_template_from_config('finalization_template')

        logger.info(f"🧠 ResponseSynthesisAgent initialized with templates from configuration")

    async def initialize(self):
        """Initialize the agent."""
        # Call base class initialization to set up LLM client
        await super().initialize()

        # Agent-specific initialization
        self.initialized = True
        logger.info(f"✅ ResponseSynthesisAgent initialized: {self.name}")


    
    def _load_template_from_config(self, template_key: str) -> 'Template':
        """Load template from YAML configuration."""
        from jinja2 import Template

        # Get template from configuration (handle both config object and dict)
        if hasattr(self.config, '__dict__'):
            template_content = getattr(self.config, template_key, None)
        else:
            template_content = self.config.get(template_key)

        if template_content:
            try:
                return Template(template_content)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {template_key} from config: {e}")
                return self._load_fallback_template_from_config(template_key)
        else:
            logger.warning(f"⚠️ No {template_key} found in config, using fallback")
            return self._load_fallback_template_from_config(template_key)
    
    def _load_fallback_template_from_config(self, template_key: str) -> 'Template':
        """Load fallback template from YAML configuration."""
        from jinja2 import Template

        # Get fallback template from configuration (handle both config object and dict)
        fallback_key = f"fallback_{template_key}"
        if hasattr(self.config, '__dict__'):
            fallback_template = getattr(self.config, fallback_key, None)
        else:
            fallback_template = self.config.get(fallback_key)

        if fallback_template:
            try:
                return Template(fallback_template)
            except Exception as e:
                logger.error(f"❌ Failed to load {fallback_key} from config: {e}")
                raise ValueError(f"No valid templates found in configuration. Please provide '{template_key}' or '{fallback_key}' in agent configuration.")
        else:
            raise ValueError(f"No templates found in configuration. Please provide '{template_key}' and '{fallback_key}' in agent configuration.")

    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process retrieved chunks and synthesize intelligent response.
        
        Args:
            input_text: JSON string containing query and retrieved chunks
            **kwargs: Additional context including conversation history
            
        Returns:
            Synthesized response as JSON string with metadata
        """
        try:
            # Parse input data
            input_data = self._parse_input_data(input_text)
            query = input_data.get('query', '')
            retrieved_chunks = input_data.get('retrieved_chunks', [])
            
            # Extract context
            context = kwargs.get('context', {})
            conversation_history = kwargs.get('conversation_history', [])
            
            logger.info(f"🔄 Starting synthesis for query: '{query[:50]}...' with {len(retrieved_chunks)} chunks")
            
            # Determine synthesis strategy based on content volume
            if self.enable_multi_turn and self._should_use_multi_turn(retrieved_chunks):
                synthesized_result = await self._multi_turn_synthesis(
                    query, retrieved_chunks, context, conversation_history
                )
            else:
                synthesized_result = await self._single_turn_synthesis(
                    query, retrieved_chunks, context, conversation_history
                )
            
            logger.info(f"✅ Response synthesized successfully")
            
            return self._format_synthesis_output(synthesized_result)
            
        except Exception as e:
            logger.error(f"❌ Response synthesis failed: {e}")
            # Graceful fallback to simple synthesis (no truncation)
            return self._create_fallback_synthesis(input_text)

    def _parse_input_data(self, input_text: str) -> Dict[str, Any]:
        """Parse input data from JSON string."""
        try:
            if input_text.strip().startswith('{'):
                return json.loads(input_text)
            else:
                # Fallback: treat as simple query
                return {
                    'query': input_text,
                    'retrieved_chunks': []
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse input data: {e}")
            return {
                'query': input_text,
                'retrieved_chunks': []
            }

    def _should_use_multi_turn(self, retrieved_chunks: List[Dict]) -> bool:
        """Determine if multi-turn synthesis is needed."""
        # Use multi-turn if we have many chunks or complex content
        total_content_length = sum(len(str(chunk.get('content', ''))) for chunk in retrieved_chunks)
        
        return (
            len(retrieved_chunks) > self.max_chunks_per_turn or
            total_content_length > self.max_response_length * 10
        )

    async def _single_turn_synthesis(self, query: str, chunks: List[Dict], 
                                   context: Dict, history: List[Dict]) -> Dict[str, Any]:
        """Perform single-turn synthesis for simpler cases."""
        # Build synthesis prompt using template
        synthesis_prompt = self._build_synthesis_prompt(
            query, chunks, context, history
        )
        
        # Call parent's LLM processing
        synthesis_result = await super().process(synthesis_prompt)
        
        # Parse and validate synthesis
        return self._parse_synthesis_result(synthesis_result, query, chunks)

    async def _multi_turn_synthesis(self, query: str, chunks: List[Dict],
                                  context: Dict, history: List[Dict]) -> Dict[str, Any]:
        """Perform multi-turn synthesis for complex responses."""
        logger.info(f"🔄 Starting multi-turn synthesis with {len(chunks)} chunks")

        # Phase 1: Analyze chunks and identify themes
        analysis_result = await self._analyze_chunks(query, chunks, context)

        # Phase 2: Synthesize sections iteratively
        synthesis_sections = []
        chunk_groups = self._group_chunks_by_theme(chunks, analysis_result)

        for i, (theme, theme_chunks) in enumerate(chunk_groups.items()):
            logger.info(f"🔄 Synthesizing section {i+1}/{len(chunk_groups)}: {theme}")

            section_result = await self._synthesize_section(
                query, theme, theme_chunks, context, synthesis_sections
            )
            synthesis_sections.append(section_result)

        # Phase 3: Finalize and create coherent response
        final_result = await self._finalize_synthesis(
            query, synthesis_sections, analysis_result, context
        )

        return final_result

    async def _analyze_chunks(self, query: str, chunks: List[Dict], context: Dict) -> Dict[str, Any]:
        """Analyze chunks to identify themes and relationships."""
        # Build analysis prompt using template
        analysis_prompt = self._build_analysis_prompt(query, chunks, context)

        # Call LLM for analysis
        analysis_result = await super().process(analysis_prompt)

        # Parse analysis result
        return self._parse_analysis_result(analysis_result)

    def _group_chunks_by_theme(self, chunks: List[Dict], analysis: Dict) -> Dict[str, List[Dict]]:
        """Group chunks by identified themes."""
        themes = analysis.get('themes', [])
        if not themes:
            # Fallback: group by similarity or create single group
            return {'main_content': chunks}

        # Simple grouping strategy - in practice, this could use embeddings or other methods
        chunk_groups = {}
        chunks_per_theme = max(1, len(chunks) // len(themes))

        for i, theme in enumerate(themes):
            start_idx = i * chunks_per_theme
            end_idx = start_idx + chunks_per_theme if i < len(themes) - 1 else len(chunks)
            chunk_groups[theme] = chunks[start_idx:end_idx]

        return chunk_groups

    async def _synthesize_section(self, query: str, theme: str, chunks: List[Dict],
                                context: Dict, previous_sections: List[Dict]) -> Dict[str, Any]:
        """Synthesize a single thematic section."""
        # Build section synthesis prompt
        section_prompt = self._build_section_prompt(
            query, theme, chunks, context, previous_sections
        )

        # Call LLM for section synthesis
        section_result = await super().process(section_prompt)

        # Parse section result
        return self._parse_section_result(section_result, theme)

    async def _finalize_synthesis(self, query: str, sections: List[Dict],
                                analysis: Dict, context: Dict) -> Dict[str, Any]:
        """Finalize synthesis by creating coherent response."""
        # Build finalization prompt using template
        finalization_prompt = self._build_finalization_prompt(
            query, sections, analysis, context
        )

        # Call LLM for finalization
        final_result = await super().process(finalization_prompt)

        # Parse and return final result
        return self._parse_finalization_result(final_result, query, sections)

    def _build_synthesis_prompt(self, query: str, chunks: List[Dict],
                              context: Dict, history: List[Dict]) -> str:
        """Build LLM prompt for single-turn synthesis using template system."""
        try:
            # Prepare template variables
            template_vars = {
                'query': query,
                'retrieved_chunks': chunks,
                'chunk_count': len(chunks),
                'max_response_length': self.max_response_length,
                'enable_contradiction_resolution': self.enable_contradiction_resolution,
                'enable_theme_analysis': self.enable_theme_analysis,
                'conversation_history': history,
                'additional_context': context,
                **self.template_variables  # Include any additional template variables from config
            }

            # Render prompt using template
            return self.synthesis_template.render(**template_vars)

        except Exception as e:
            logger.error(f"❌ Synthesis template rendering failed: {e}")
            return self._create_emergency_prompt(query, chunks, 'synthesis')

    def _build_analysis_prompt(self, query: str, chunks: List[Dict], context: Dict) -> str:
        """Build LLM prompt for chunk analysis using template system."""
        try:
            # Prepare template variables
            template_vars = {
                'query': query,
                'retrieved_chunks': chunks,
                'chunk_count': len(chunks),
                'enable_theme_analysis': self.enable_theme_analysis,
                'enable_contradiction_resolution': self.enable_contradiction_resolution,
                'additional_context': context,
                **self.template_variables
            }

            # Render prompt using template
            return self.analysis_template.render(**template_vars)

        except Exception as e:
            logger.error(f"❌ Analysis template rendering failed: {e}")
            return self._create_emergency_prompt(query, chunks, 'analysis')

    def _build_section_prompt(self, query: str, theme: str, chunks: List[Dict],
                            context: Dict, previous_sections: List[Dict]) -> str:
        """Build LLM prompt for section synthesis using template system."""
        try:
            # Prepare template variables
            template_vars = {
                'query': query,
                'theme': theme,
                'theme_chunks': chunks,
                'chunk_count': len(chunks),
                'previous_sections': previous_sections,
                'section_number': len(previous_sections) + 1,
                'additional_context': context,
                **self.template_variables
            }

            # Render prompt using template
            return self.continuation_template.render(**template_vars)

        except Exception as e:
            logger.error(f"❌ Section template rendering failed: {e}")
            return self._create_emergency_prompt(query, chunks, 'section')

    def _build_finalization_prompt(self, query: str, sections: List[Dict],
                                 analysis: Dict, context: Dict) -> str:
        """Build LLM prompt for response finalization using template system."""
        try:
            # Prepare template variables
            template_vars = {
                'query': query,
                'synthesis_sections': sections,
                'section_count': len(sections),
                'analysis_result': analysis,
                'max_response_length': self.max_response_length,
                'additional_context': context,
                **self.template_variables
            }

            # Render prompt using template
            return self.finalization_template.render(**template_vars)

        except Exception as e:
            logger.error(f"❌ Finalization template rendering failed: {e}")
            return self._create_emergency_prompt(query, sections, 'finalization')

    def _create_emergency_prompt(self, query: str, data: Any, prompt_type: str) -> str:
        """Create emergency prompt when all template systems fail."""
        # Use only configuration-driven content, no hardcoded prompts
        emergency_key = f'emergency_{prompt_type}_instruction'
        if hasattr(self.config, '__dict__'):
            emergency_instruction = getattr(self.config, emergency_key, None)
        else:
            emergency_instruction = self.config.get(emergency_key)

        if not emergency_instruction:
            raise ValueError(f"No {emergency_key} found in configuration. All template systems failed and no fallback instruction provided.")

        return f"{emergency_instruction}: {query}"

    def _parse_synthesis_result(self, result: str, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Parse LLM synthesis result with fallback handling."""
        try:
            # Try to parse JSON response
            if result.strip().startswith('{'):
                parsed = json.loads(result)
                return {
                    'synthesized_response': parsed.get('synthesized_response', result),
                    'key_themes': parsed.get('key_themes', []),
                    'source_integration': parsed.get('source_integration', {}),
                    'confidence_score': parsed.get('confidence_score', 0.8),
                    'synthesis_metadata': parsed.get('synthesis_metadata', {})
                }
            else:
                # Fallback: treat entire response as synthesized content (no truncation)
                return {
                    'synthesized_response': result.strip(),
                    'key_themes': [],
                    'source_integration': {},
                    'confidence_score': 0.7,
                    'synthesis_metadata': {'method': 'single_turn_fallback'}
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse synthesis result: {e}")
            return self._create_fallback_synthesis_result(query, chunks)

    def _parse_analysis_result(self, result: str) -> Dict[str, Any]:
        """Parse chunk analysis result."""
        try:
            if result.strip().startswith('{'):
                return json.loads(result)
            else:
                return {
                    'themes': ['main_content'],
                    'relationships': [],
                    'contradictions': [],
                    'key_concepts': []
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse analysis result: {e}")
            return {'themes': ['main_content'], 'relationships': [], 'contradictions': [], 'key_concepts': []}

    def _parse_section_result(self, result: str, theme: str) -> Dict[str, Any]:
        """Parse section synthesis result."""
        try:
            if result.strip().startswith('{'):
                parsed = json.loads(result)
                return {
                    'theme': theme,
                    'content': parsed.get('content', result),
                    'key_points': parsed.get('key_points', []),
                    'sources_used': parsed.get('sources_used', [])
                }
            else:
                return {
                    'theme': theme,
                    'content': result.strip(),
                    'key_points': [],
                    'sources_used': []
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse section result: {e}")
            return {'theme': theme, 'content': result, 'key_points': [], 'sources_used': []}

    def _parse_finalization_result(self, result: str, query: str, sections: List[Dict]) -> Dict[str, Any]:
        """Parse finalization result."""
        try:
            if result.strip().startswith('{'):
                parsed = json.loads(result)
                return {
                    'synthesized_response': parsed.get('final_response', result),
                    'key_themes': [section.get('theme', '') for section in sections],
                    'source_integration': parsed.get('source_integration', {}),
                    'confidence_score': parsed.get('confidence_score', 0.8),
                    'synthesis_metadata': {
                        'method': 'multi_turn',
                        'sections_count': len(sections),
                        **parsed.get('metadata', {})
                    }
                }
            else:
                return {
                    'synthesized_response': result.strip()[:self.max_response_length],
                    'key_themes': [section.get('theme', '') for section in sections],
                    'source_integration': {},
                    'confidence_score': 0.7,
                    'synthesis_metadata': {'method': 'multi_turn_fallback', 'sections_count': len(sections)}
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse finalization result: {e}")
            return self._create_fallback_synthesis_result(query, [])

    def _create_fallback_synthesis_result(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Create fallback synthesis result when parsing fails."""
        return {
            'synthesized_response': f"Response to: {query}\n\nBased on {len(chunks)} retrieved sources, here is a synthesized response.",
            'key_themes': ['general_response'],
            'source_integration': {'sources_count': len(chunks)},
            'confidence_score': 0.5,
            'synthesis_metadata': {'method': 'fallback', 'error': 'parsing_failed'}
        }

    def _create_fallback_synthesis(self, input_text: str) -> str:
        """Create fallback synthesis when processing fails."""
        fallback_result = {
            'synthesized_response': f"Fallback response for input: {input_text[:100]}...",
            'key_themes': ['fallback'],
            'source_integration': {},
            'confidence_score': 0.3,
            'synthesis_metadata': {'method': 'emergency_fallback', 'error': 'processing_failed'}
        }
        return json.dumps(fallback_result)

    def _format_synthesis_output(self, synthesis_result: Dict[str, Any]) -> str:
        """Format synthesis result as JSON string for AgentStep compatibility."""
        try:
            return json.dumps(synthesis_result, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to format synthesis output: {e}")
            return json.dumps({
                'synthesized_response': 'Error formatting response',
                'error': str(e)
            })
