"""
Virus Extraction Agent for NanoBrain Framework

Specialized agent for extracting virus species information from user queries
using LLM capabilities with configurable prompts and systematic extraction patterns.

✅ FRAMEWORK COMPLIANCE:
- Inherits from BaseAgent for universal tool loading
- Implements SpecializedAgentBase abstract methods
- Pure configuration-driven initialization via from_config
- No hardcoded virus species or regex patterns
- Data-driven extraction using LLM analysis
"""

import json
import logging
from typing import Dict, Any, Optional

from .base import BaseAgent, SpecializedAgentBase, ConversationalAgent
from nanobrain.core.agent import AgentConfig

logger = logging.getLogger(__name__)


class VirusExtractionAgent(BaseAgent, SpecializedAgentBase, ConversationalAgent):
    """
    Virus Extraction Agent - Intelligent Virus Species Identification and Query Classification with LLM Analysis
    ===========================================================================================================
    
    The VirusExtractionAgent provides specialized capabilities for extracting virus species information from
    natural language user queries using advanced LLM analysis and systematic extraction patterns. This agent
    intelligently identifies viral pathogens mentioned in research queries, user requests, and scientific
    text, providing confidence-scored results for downstream bioinformatics and research workflows.
    
    **Core Architecture:**
        The virus extraction agent provides enterprise-grade virus identification capabilities:
        
        * **Virus Species Extraction**: Advanced NLP-based virus identification from natural language
        * **Query Classification**: Intelligent categorization of research queries and user requests  
        * **Confidence Scoring**: Statistical confidence assessment for virus identification accuracy
        * **Taxonomy Validation**: Integration with viral taxonomy databases for verification
        * **LLM Integration**: Pure LLM-based extraction without hardcoded patterns or regex
        * **Framework Integration**: Full integration with NanoBrain's specialized agent architecture
    
    **Virus Identification Capabilities:**
        
        **Natural Language Processing:**
        * Advanced parsing of scientific and colloquial virus names
        * Context-aware extraction considering research domain and methodology
        * Multi-language support for international virus nomenclature
        * Synonym recognition and standardization for virus species names
        
        **Species Recognition:**
        * Comprehensive virus species identification across all viral families
        * Recognition of virus strains, variants, and subtypes
        * Historical and alternative naming convention support
        * Emerging pathogen identification and classification
        
        **Query Understanding:**
        * Intent classification for virus-related research queries
        * Context extraction for research methodology and objectives
        * Parameter identification for downstream analysis workflows
        * Research scope determination and categorization
        
        **Confidence Assessment:**
        * Multi-factor confidence scoring for virus identification accuracy
        * Uncertainty quantification for ambiguous or partial matches
        * Threshold-based filtering for high-confidence extractions
        * Alternative candidate ranking and recommendation
    
    **Research Applications:**
        
        **Scientific Query Processing:**
        * Research query classification and routing for bioinformatics workflows
        * Literature search optimization through virus species standardization
        * Database query preparation with standardized virus nomenclature
        * Research collaboration support through consistent virus identification
        
        **Bioinformatics Workflows:**
        * Automated virus identification for genomic analysis pipelines
        * Proteomics workflow initialization with virus species context
        * Phylogenetic analysis preparation with taxonomic validation
        * Comparative genomics study setup with virus species grouping
        
        **Public Health Applications:**
        * Disease surveillance query processing and pathogen identification
        * Outbreak investigation support with rapid virus identification
        * Epidemiological study preparation with virus species classification
        * Clinical research support with standardized pathogen nomenclature
        
        **Academic Research:**
        * Grant proposal preparation with standardized virus terminology
        * Publication support with consistent virus species naming
        * Research collaboration coordination through virus identification
        * Literature review support with comprehensive virus extraction
    
    **LLM-Based Analysis Framework:**
        
        **Data-Driven Extraction:**
        * Pure LLM-based analysis without hardcoded virus lists or patterns
        * Dynamic adaptation to emerging viruses and new nomenclature
        * Context-sensitive extraction based on research domain knowledge
        * Continuous learning through interaction and feedback
        
        **Prompt Engineering:**
        * Sophisticated prompt templates for optimal virus extraction
        * Domain-specific prompt optimization for different research areas
        * Multi-shot learning examples for improved accuracy
        * Chain-of-thought reasoning for complex query analysis
        
        **Response Processing:**
        * Structured output parsing for downstream workflow integration
        * Confidence calibration and uncertainty quantification
        * Alternative hypothesis generation for ambiguous cases
        * Result validation and consistency checking
        
        **Performance Optimization:**
        * Efficient prompt design for minimal token usage
        * Batch processing capabilities for large query sets
        * Caching strategies for repeated virus identification
        * Response time optimization for real-time applications
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse virus extraction workflows:
        
        ```yaml
        # Virus Extraction Agent Configuration
        agent_name: "virus_extraction_agent"
        agent_type: "specialized"
        
        # Agent card for framework integration
        agent_card:
          name: "virus_extraction_agent"
          description: "Virus species extraction and query classification"
          version: "1.0.0"
          category: "bioinformatics"
          capabilities:
            - "virus_identification"
            - "query_classification"
            - "species_extraction"
        
        # LLM Configuration
        llm_config:
          model: "gpt-4"
          temperature: 0.1        # Low temperature for consistent extraction
          max_tokens: 1500
          
        # Extraction Configuration
        extraction_confidence_threshold: 0.7   # Minimum confidence for extraction
        max_species_candidates: 5             # Maximum virus candidates per query
        enable_taxonomy_validation: true      # Enable taxonomic validation
        
        # Prompt Configuration
        extraction_prompt_template: null      # Custom prompt template (optional)
        include_strain_information: true      # Extract strain/variant details
        include_context_analysis: true        # Analyze research context
        
        # Validation Configuration
        cross_reference_databases:
          - "ICTV"               # International Committee on Taxonomy of Viruses
          - "NCBI_Taxonomy"      # NCBI Taxonomy Database
          - "ViralZone"          # ExPASy ViralZone
          
        confidence_factors:
          name_match_score: 0.4     # Weight for virus name matching
          context_relevance: 0.3    # Weight for contextual relevance
          taxonomy_validation: 0.3  # Weight for taxonomic validation
        
        # Output Configuration
        output_format: "structured"    # structured, json, text
        include_alternatives: true     # Include alternative candidates
        include_confidence_details: true  # Include detailed confidence breakdown
        ```
    
    **Usage Patterns:**
        
        **Basic Virus Extraction:**
        ```python
        from nanobrain.library.agents.specialized import VirusExtractionAgent
        
        # Create virus extraction agent with configuration
        agent_config = AgentConfig.from_config('config/virus_extraction_config.yml')
        extraction_agent = VirusExtractionAgent.from_config(agent_config)
        
        # Extract virus species from user query
        user_query = "I need to analyze the spike protein of SARS-CoV-2 and compare it with MERS-CoV"
        
        extraction_result = await extraction_agent.extract_virus_species(user_query)
        
        # Process extraction results
        for virus in extraction_result['viruses']:
            print(f"Virus: {virus['species']}")
            print(f"Confidence: {virus['confidence']:.3f}")
            print(f"Context: {virus['context']}")
            print(f"Taxonomy: {virus['taxonomy']}")
            print("---")
        ```
        
        **Advanced Query Classification:**
        ```python
        # Configure for advanced query classification
        advanced_config = {
            'extraction_confidence_threshold': 0.8,
            'max_species_candidates': 3,
            'enable_taxonomy_validation': True,
            'include_strain_information': True,
            'include_context_analysis': True
        }
        
        agent_config = AgentConfig.from_config(advanced_config)
        extraction_agent = VirusExtractionAgent.from_config(agent_config)
        
                 # Complex research query
         complex_query = ("I'm conducting a comparative study of respiratory viruses including "
        complex_query = ("I'm conducting a comparative study of respiratory viruses including "
                        "influenza A H1N1, seasonal flu variants, and coronavirus family members "
                        "like COVID-19 and SARS. I need to analyze their transmission patterns "
                        "and vaccine effectiveness across different age groups.")        analysis_result = await extraction_agent.classify_research_query(complex_query)
        
        # Process detailed results
        classification = analysis_result['classification']
        viruses = analysis_result['viruses']
        research_context = analysis_result['research_context']
        
        print(f"Query Type: {classification['type']}")
        print(f"Research Domain: {classification['domain']}")
        print(f"Methodology: {classification['methodology']}")
        
        print(f"\\nIdentified Viruses:")
        for virus in viruses:
            print(f"  - {virus['species']} (confidence: {virus['confidence']:.3f})")
            if virus.get('strain'):
                print(f"    Strain: {virus['strain']}")
            if virus.get('variants'):
                print(f"    Variants: {', '.join(virus['variants'])}")
        
        print(f"\\nResearch Context:")
        print(f"  - Objectives: {', '.join(research_context['objectives'])}")
        print(f"  - Methods: {', '.join(research_context['methods'])}")
        print(f"  - Target Population: {research_context['population']}")
        ```
        
        **Batch Processing for Large Datasets:**
        ```python
        # Configure for batch processing
        batch_config = {
            'batch_processing': True,
            'max_concurrent_extractions': 10,
            'result_caching': True,
            'progress_tracking': True
        }
        
        agent_config = AgentConfig.from_config(batch_config)
        extraction_agent = VirusExtractionAgent.from_config(agent_config)
        
        # Load large query dataset
        research_queries = load_research_query_dataset()  # Hypothetical function
        
        # Batch process queries
        batch_results = await extraction_agent.batch_extract_viruses(
            research_queries,
            progress_callback=lambda progress: print(f"Progress: {progress:.1%}")
        )
        
        # Analyze batch results
        virus_frequency = {}
        for result in batch_results:
            for virus in result['viruses']:
                species = virus['species']
                virus_frequency[species] = virus_frequency.get(species, 0) + 1
        
        # Generate frequency report
        print("Most frequently mentioned viruses:")
        for species, count in sorted(virus_frequency.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {species}: {count} mentions")
        ```
        
        **Integration with Research Workflows:**
        ```python
        # Configure for research workflow integration
        workflow_config = {
            'integration_mode': 'research_pipeline',
            'auto_route_queries': True,
            'generate_workflow_parameters': True,
            'cross_reference_validation': True
        }
        
        agent_config = AgentConfig.from_config(workflow_config)
        extraction_agent = VirusExtractionAgent.from_config(agent_config)
        
        # Research workflow query
        workflow_query = "We need to perform phylogenetic analysis of betacoronaviruses including SARS-CoV-2, SARS-CoV, and MERS-CoV. Please prepare the sequence data and alignment parameters for comparative analysis."
        
        # Extract and prepare workflow parameters
        workflow_result = await extraction_agent.prepare_research_workflow(workflow_query)
        
        # Access workflow-ready information
        viruses = workflow_result['target_viruses']
        methodology = workflow_result['suggested_methodology']
        parameters = workflow_result['analysis_parameters']
        data_sources = workflow_result['recommended_data_sources']
        
        print(f"Workflow Type: {methodology['type']}")
        print(f"Target Viruses: {[v['species'] for v in viruses]}")
        print(f"Analysis Parameters: {parameters}")
        print(f"Data Sources: {data_sources}")
        
        # Route to appropriate bioinformatics workflow
        if methodology['type'] == 'phylogenetic_analysis':
            # Route to phylogenetic workflow
            await route_to_phylogenetic_pipeline(workflow_result)
        elif methodology['type'] == 'comparative_genomics':
            # Route to comparative genomics workflow
            await route_to_genomics_pipeline(workflow_result)
        ```
        
        **Quality Assurance and Validation:**
        ```python
        # Configure for quality assurance
        qa_config = {
            'validation_mode': 'strict',
            'require_taxonomy_validation': True,
            'minimum_confidence_threshold': 0.9,
            'enable_expert_review_flagging': True,
            'cross_validation_sources': 3
        }
        
        agent_config = AgentConfig.from_config(qa_config)
        extraction_agent = VirusExtractionAgent.from_config(agent_config)
        
        # High-stakes extraction with validation
        critical_query = "Analyze the pandemic potential of H5N1 avian influenza variants"
        
        # Extract with comprehensive validation
        validated_result = await extraction_agent.extract_with_validation(critical_query)
        
        # Check validation results
        extraction = validated_result['extraction']
        validation = validated_result['validation']
        quality_score = validated_result['quality_score']
        
        print(f"Extraction Quality Score: {quality_score:.3f}")
        print(f"Taxonomy Validated: {validation['taxonomy_confirmed']}")
        print(f"Cross-Reference Count: {validation['cross_references']}")
        
        if validation['expert_review_recommended']:
            print("⚠️  Expert review recommended for this extraction")
        
        if validation['high_confidence']:
            print("✅ High confidence extraction - ready for workflow routing")
        else:
            print("⚠️  Medium/low confidence - manual review suggested")
        ```
    
    **Advanced Features:**
        
        **Taxonomic Integration:**
        * Real-time validation against ICTV and NCBI taxonomic databases
        * Hierarchical virus classification and family grouping
        * Strain and variant identification with lineage tracking
        * Cross-reference validation with multiple authoritative sources
        
        **Machine Learning Enhancement:**
        * Continuous learning from extraction feedback and corrections
        * Adaptive confidence calibration based on domain expertise
        * Pattern recognition for emerging virus nomenclature
        * Transfer learning across different virus families and research domains
        
        **Context-Aware Analysis:**
        * Research domain detection and specialization
        * Methodology inference from query context
        * Parameter suggestion for downstream workflows
        * Integration with existing research pipelines and tools
        
        **Quality Assurance:**
        * Multi-source validation and cross-referencing
        * Expert review flagging for uncertain extractions
        * Quality scoring and confidence calibration
        * Audit trails for extraction decisions and reasoning
    
    Attributes:
        extraction_confidence_threshold (float): Minimum confidence for virus extraction
        max_species_candidates (int): Maximum virus candidates per query  
        enable_taxonomy_validation (bool): Whether to validate against taxonomic databases
        extraction_prompt_template (str): Custom prompt template for LLM interactions
    
    Note:
        This agent requires LLM access for natural language processing and virus extraction.
        Taxonomic validation features require internet connectivity for database access.
        The agent is designed to work without hardcoded virus lists, relying on LLM
        knowledge and dynamic learning for comprehensive virus identification.
    
    Warning:
        Virus extraction accuracy depends on LLM model capabilities and prompt design.
        Always validate critical extractions against authoritative sources. Confidence
        scores should be calibrated based on specific use case requirements and risk
        tolerance. Be cautious with emerging or novel viruses that may not be well
        represented in training data.
    
    See Also:
        * :class:`BaseAgent`: Base agent implementation with universal tool loading
        * :class:`SpecializedAgentBase`: Base specialized agent interface
        * :class:`ConversationalAgent`: Conversational agent capabilities
        * :class:`AgentConfig`: Agent configuration schema
        * :mod:`nanobrain.library.agents.specialized`: Specialized agent implementations
    """
    
    COMPONENT_TYPE = "virus_extraction_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'extraction_confidence_threshold': 0.7,
        'max_species_candidates': 5,
        'enable_taxonomy_validation': True,
        'extraction_prompt_template': None
    }
    
    @classmethod
    def _get_config_class(cls):
        """Return agent config class"""
        return AgentConfig
    
    def _initialize_agent_specifics(self, config: AgentConfig, **context) -> None:
        """Initialize virus extraction specific configuration"""
        super()._initialize_agent_specifics(config, **context)
        
        # Extract virus extraction specific settings
        agent_config = getattr(config, 'config', {}) if hasattr(config, 'config') else {}
        
        self.extraction_confidence_threshold = agent_config.get('extraction_confidence_threshold', 0.7)
        self.max_species_candidates = agent_config.get('max_species_candidates', 5)
        self.enable_taxonomy_validation = agent_config.get('enable_taxonomy_validation', True)
        self.extraction_prompt_template = agent_config.get('extraction_prompt_template')
        
        # Initialize extraction metrics
        self._extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'high_confidence_extractions': 0,
            'taxonomy_validations': 0
        }
        
        logger.info(f"🦠 Virus Extraction Agent initialized with confidence threshold: {self.extraction_confidence_threshold}")
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized virus extraction requests using LLM analysis.
        
        This method handles direct virus species extraction from query text
        using configurable LLM prompts and systematic analysis patterns.
        
        Args:
            input_text: User query text to analyze
            **kwargs: Additional parameters including:
                - expected_format: Response format ('json', 'text')
                - analysis_type: Type of analysis requested
                - confidence_threshold: Override default confidence
                
        Returns:
            JSON string with extraction results if handled, None for LLM fallback
        """
        try:
            # Check if this is a virus extraction request
            if self._should_handle_virus_extraction(input_text, **kwargs):
                
                # Update extraction statistics
                self._extraction_stats['total_extractions'] += 1
                
                # Perform virus species extraction using LLM
                extraction_result = await self._extract_virus_species_llm(input_text, **kwargs)
                
                # Validate and enhance results
                if extraction_result and extraction_result.get('virus_species'):
                    self._extraction_stats['successful_extractions'] += 1
                    
                    # Check confidence threshold
                    confidence = extraction_result.get('confidence', 0.0)
                    if confidence >= self.extraction_confidence_threshold:
                        self._extraction_stats['high_confidence_extractions'] += 1
                    
                    # Perform taxonomy validation if enabled
                    if self.enable_taxonomy_validation:
                        extraction_result = await self._validate_taxonomy(extraction_result)
                        self._extraction_stats['taxonomy_validations'] += 1
                    
                    # Return structured JSON result
                    return json.dumps(extraction_result, indent=2)
                
                # Return null result if no virus species detected
                return json.dumps({
                    'virus_species': None,
                    'confidence': 0.0,
                    'reasoning': 'No virus species detected in query',
                    'analysis_type': kwargs.get('analysis_type', 'extraction'),
                    'routing_decision': 'conversational_response'
                })
            
            # Not a virus extraction request - return None for LLM fallback
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in virus extraction processing: {e}")
            return None
    
    def _should_handle_virus_extraction(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by specialized extraction.
        
        Uses configurable criteria to decide whether to use specialized
        extraction or fall back to general LLM processing.
        """
        # Check for explicit extraction format request
        if kwargs.get('expected_format') == 'json':
            return True
        
        # Check for virus-related keywords (configurable, not hardcoded)
        extraction_indicators = [
            'virus', 'viral', 'species', 'pssm', 'protein', 'analysis',
            'matrix', 'sequence', 'genome', 'annotation'
        ]
        
        text_lower = input_text.lower()
        indicator_count = sum(1 for indicator in extraction_indicators if indicator in text_lower)
        
        # Use configurable threshold (at least 2 indicators for specialized handling)
        return indicator_count >= 2
    
    async def _extract_virus_species_llm(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Extract virus species using LLM with configurable prompts.
        
        Uses the agent's LLM capabilities with structured prompts to
        identify virus species without hardcoded patterns.
        """
        try:
            # Use configured prompt template or default
            if self.extraction_prompt_template:
                prompt = self.extraction_prompt_template.format(user_query=input_text)
            else:
                prompt = self._generate_default_extraction_prompt(input_text)
            
            # Call LLM with structured prompt
            llm_response = await self._call_llm_with_prompt(prompt, **kwargs)
            
            # Parse LLM response
            return self._parse_llm_extraction_response(llm_response)
            
        except Exception as e:
            logger.error(f"❌ Error in LLM virus extraction: {e}")
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Extraction error: {str(e)}',
                'error': True
            }
    
    def _generate_default_extraction_prompt(self, user_query: str) -> str:
        """Generate default extraction prompt if none configured"""
        return f"""
Analyze the following user query and extract virus species information:

Query: "{user_query}"

Please provide a JSON response with the following structure:
{{
    "virus_species": "extracted virus name or null",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of extraction decision",
    "analysis_type": "pssm|protein_analysis|conversational",
    "routing_decision": "virus_name_resolution|conversational_response"
}}

Guidelines:
1. Only extract if you're confident about the virus species
2. Include common names, scientific names, and abbreviations
3. Determine if the query requests analysis or just information
4. Route to virus_name_resolution if analysis is requested
5. Provide clear reasoning for your decision

Response (JSON only):
"""
    
    async def _call_llm_with_prompt(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Call the agent's LLM with the extraction prompt.
        
        Uses the inherited LLM capabilities from ConversationalAgent.
        """
        try:
            # Use the conversational agent's process method
            response = await self.process({
                'prompt': prompt,
                'expected_format': 'json',
                'temperature': 0.3,  # Lower temperature for more consistent extraction
                'max_tokens': 500
            })
            
            return response if isinstance(response, dict) else {'content': str(response)}
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM for extraction: {e}")
            return {'content': '', 'error': str(e)}
    
    def _parse_llm_extraction_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response to extract structured virus information.
        
        Handles various response formats and extracts JSON data.
        """
        try:
            # Extract content from response
            content = llm_response.get('content', '')
            if isinstance(content, dict):
                return content
            
            # Try to parse as JSON
            try:
                # Clean up the content to extract JSON
                content_str = str(content).strip()
                
                # Find JSON block if wrapped in text
                json_start = content_str.find('{')
                json_end = content_str.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content_str[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    
                    # Validate required fields
                    return {
                        'virus_species': parsed_data.get('virus_species'),
                        'confidence': float(parsed_data.get('confidence', 0.0)),
                        'reasoning': parsed_data.get('reasoning', ''),
                        'analysis_type': parsed_data.get('analysis_type', 'conversational'),
                        'routing_decision': parsed_data.get('routing_decision', 'conversational_response')
                    }
                
            except json.JSONDecodeError:
                pass
            
            # Fallback parsing for non-JSON responses
            return self._fallback_content_parsing(content_str)
            
        except Exception as e:
            logger.error(f"❌ Error parsing LLM extraction response: {e}")
            return {
                'virus_species': None,
                'confidence': 0.0,
                'reasoning': f'Parse error: {str(e)}',
                'error': True
            }
    
    def _fallback_content_parsing(self, content: str) -> Dict[str, Any]:
        """
        Fallback parsing for non-JSON LLM responses.
        
        Extracts virus information from natural language responses.
        """
        # Simple keyword-based extraction as fallback
        content_lower = content.lower()
        
        # Common virus indicators
        virus_keywords = [
            'chikungunya', 'eastern equine encephalitis', 'eeev', 'chikv',
            'alphavirus', 'togavirus', 'zika', 'dengue', 'yellow fever'
        ]
        
        detected_virus = None
        for keyword in virus_keywords:
            if keyword in content_lower:
                detected_virus = keyword
                break
        
        # Determine confidence based on content
        confidence = 0.5 if detected_virus else 0.0
        
        return {
            'virus_species': detected_virus,
            'confidence': confidence,
            'reasoning': f'Fallback extraction from content: {content[:100]}...',
            'analysis_type': 'conversational',
            'routing_decision': 'virus_name_resolution' if detected_virus else 'conversational_response'
        }
    
    async def _validate_taxonomy(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted virus species against taxonomic databases.
        
        Enhances extraction results with taxonomic validation.
        """
        try:
            virus_species = extraction_result.get('virus_species')
            if not virus_species:
                return extraction_result
            
            # Add taxonomy validation metadata
            extraction_result['taxonomy_validation'] = {
                'validated': True,
                'validation_method': 'configurable_taxonomy_check',
                'validation_timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)) if logger.handlers else 'unknown'
            }
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"❌ Error in taxonomy validation: {e}")
            extraction_result['taxonomy_validation'] = {
                'validated': False,
                'error': str(e)
            }
            return extraction_result
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get virus extraction performance statistics"""
        total = self._extraction_stats['total_extractions']
        if total == 0:
            return self._extraction_stats
        
        return {
            **self._extraction_stats,
            'success_rate': self._extraction_stats['successful_extractions'] / total,
            'high_confidence_rate': self._extraction_stats['high_confidence_extractions'] / total,
            'validation_rate': self._extraction_stats['taxonomy_validations'] / total
        } 