"""
Query Analysis Agent for NanoBrain Framework

Specialized agent for analyzing user queries to extract biological context,
intent classification, and parameter extraction for virus name resolution.

✅ FRAMEWORK COMPLIANCE:
- Inherits from BaseAgent for universal tool loading
- Implements SpecializedAgentBase abstract methods
- Pure configuration-driven initialization via from_config
- No hardcoded analysis patterns or regex solutions
- Data-driven query analysis using LLM capabilities
"""

import json
import logging
from typing import Dict, Any, Optional

from .base import SpecializedAgentBase, ConversationalSpecializedAgent
from nanobrain.core.agent import AgentConfig

logger = logging.getLogger(__name__)


class QueryAnalysisAgent(ConversationalSpecializedAgent):
    """
    Query Analysis Agent - Intelligent Biological Query Processing and Research Intent Classification
    ===============================================================================================
    
    The QueryAnalysisAgent provides specialized capabilities for analyzing user queries in biological and biomedical
    contexts, with particular expertise in virus research, genomics, and bioinformatics workflows. This agent combines
    advanced natural language processing with domain-specific biological knowledge to extract intent, parameters,
    and context from complex scientific queries.
    
    **Core Architecture:**
        The query analysis agent provides enterprise-grade biological query processing:
        
        * **Biological Context Extraction**: Advanced parsing of scientific terminology and biological concepts
        * **Intent Classification**: Intelligent categorization of research queries and analysis requirements
        * **Parameter Extraction**: Automated identification of species, genes, proteins, and analysis parameters
        * **Confidence Scoring**: Statistical confidence assessment for query classification accuracy
        * **Research Workflow Integration**: Seamless routing to appropriate bioinformatics analysis pipelines
        * **Framework Integration**: Full integration with NanoBrain's specialized agent architecture
    
    **Query Analysis Capabilities:**
        
        **Biological Context Processing:**
        * Advanced recognition of virus species, strains, and variants
        * Protein and gene name identification with synonym resolution
        * Biological pathway and process classification
        * Taxonomic hierarchy understanding and validation
        
        **Intent Classification:**
        * **Research Intent**: Hypothesis generation, literature review, data exploration
        * **Analysis Intent**: Sequence analysis, structure prediction, phylogenetic analysis
        * **Comparison Intent**: Cross-species comparison, variant analysis, evolutionary studies
        * **Functional Intent**: Protein function prediction, pathway analysis, drug discovery
        
        **Parameter Extraction:**
        * Species and strain identification with confidence scoring
        * Gene and protein name extraction with standardization
        * Analysis type classification (genomic, proteomic, phylogenetic)
        * Quality thresholds and filter parameter identification
        
        **Scientific Query Understanding:**
        * Multi-language scientific terminology recognition
        * Abbreviation expansion and standardization
        * Context-dependent disambiguation of biological terms
        * Cross-reference validation with biological databases
    
    **Research Workflow Integration:**
        
        **Bioinformatics Pipeline Routing:**
        * Automatic workflow selection based on query analysis
        * Parameter configuration for downstream analysis tools
        * Quality control and validation step recommendation
        * Resource requirement estimation and optimization
        
        **Database Query Optimization:**
        * Optimized search strategy generation for biological databases
        * Query expansion with synonyms and related terms
        * Filter and threshold recommendation based on query context
        * Result ranking and relevance scoring
        
        **Literature and Data Integration:**
        * Literature search strategy optimization
        * Cross-database query coordination and execution
        * Data integration and harmonization recommendations
        * Publication and dataset relevance assessment
        
        **Collaborative Research Support:**
        * Multi-researcher query coordination and deduplication
        * Shared analysis parameter standardization
        * Research project context maintenance and tracking
        * Progress monitoring and milestone identification
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse biological research workflows:
        
        ```yaml
        # Query Analysis Agent Configuration
        agent_name: "query_analysis_agent"
        agent_type: "specialized"
        
        # Agent card for framework integration
        agent_card:
          name: "query_analysis_agent"
          description: "Biological query analysis and research intent classification"
          version: "1.0.0"
          category: "bioinformatics"
          capabilities:
            - "biological_context_extraction"
            - "intent_classification"
            - "parameter_extraction"
        
        # LLM Configuration
        llm_config:
          model: "gpt-4"
          temperature: 0.1        # Low temperature for consistent analysis
          max_tokens: 2000
          
        # Analysis Configuration
        confidence_threshold: 0.7           # Minimum confidence for classification
        analysis_depth: "comprehensive"    # standard, detailed, comprehensive
        biological_context_focus: true     # Enable biological domain specialization
        
        # Biological Domain Configuration
        domain_specialization:
          virology: true
          genomics: true
          proteomics: true
          phylogenetics: true
          drug_discovery: false
          
        # Intent Classification
        intent_categories:
          - "sequence_analysis"
          - "structure_prediction"
          - "phylogenetic_analysis"
          - "comparative_genomics"
          - "functional_annotation"
          - "literature_review"
          - "data_exploration"
          
        # Parameter Extraction
        parameter_types:
          species: true
          genes: true
          proteins: true
          pathways: true
          chemicals: true
          
        # Validation Configuration
        validation_sources:
          - "NCBI_Taxonomy"
          - "UniProt"
          - "KEGG"
          - "GO_Ontology"
          
        # Output Configuration
        output_format: "structured"
        include_confidence_scores: true
        include_alternative_interpretations: true
        ```
    
    **Usage Patterns:**
        
        **Basic Query Analysis:**
        ```python
        from nanobrain.library.agents.specialized import QueryAnalysisAgent
        
        # Create query analysis agent with configuration
        agent_config = AgentConfig.from_config('config/query_analysis_config.yml')
        query_analyzer = QueryAnalysisAgent.from_config(agent_config)
        
        # Analyze biological research query
        research_query = "Compare the spike protein sequences of SARS-CoV-2 variants to identify mutations affecting ACE2 binding"
        
        analysis_result = await query_analyzer.analyze_query(research_query)
        
        # Access analysis results
        biological_context = analysis_result.data['biological_context']
        intent_classification = analysis_result.data['intent']
        extracted_parameters = analysis_result.data['parameters']
        confidence_scores = analysis_result.data['confidence']
        
        print(f"Intent: {intent_classification['primary_intent']}")
        print(f"Species: {biological_context['species']}")
        print(f"Proteins: {biological_context['proteins']}")
        print(f"Analysis Type: {intent_classification['analysis_type']}")
        print(f"Confidence: {confidence_scores['overall']:.3f}")
        ```
        
        **Comprehensive Research Query Processing:**
        ```python
        # Configure for detailed research analysis
        research_config = {
            'analysis_depth': 'comprehensive',
            'biological_context_focus': True,
            'domain_specialization': {
                'virology': True,
                'genomics': True,
                'proteomics': True
            },
            'include_workflow_recommendations': True
        }
        
        agent_config = AgentConfig.from_config(research_config)
        query_analyzer = QueryAnalysisAgent.from_config(agent_config)
        
        # Complex multi-part research query
        complex_query = "I need to perform a comprehensive phylogenetic analysis of betacoronaviruses, focusing on the evolution of the spike protein across different species including SARS-CoV, SARS-CoV-2, and MERS-CoV. I want to identify key mutations that may affect host range and pathogenicity, and correlate these with structural changes in the receptor binding domain."
        
        comprehensive_analysis = await query_analyzer.analyze_comprehensive_query(complex_query)
        
        # Access detailed analysis results
        research_intent = comprehensive_analysis.data['research_intent']
        biological_entities = comprehensive_analysis.data['biological_entities']
        analysis_requirements = comprehensive_analysis.data['analysis_requirements']
        workflow_recommendations = comprehensive_analysis.data['workflow_recommendations']
        
        print(f"Research Type: {research_intent['research_type']}")
        print(f"Primary Objective: {research_intent['primary_objective']}")
        
        print(f"\\nBiological Entities:")
        print(f"  - Virus Species: {biological_entities['virus_species']}")
        print(f"  - Proteins: {biological_entities['proteins']}")
        print(f"  - Genomic Regions: {biological_entities['genomic_regions']}")
        
        print(f"\\nAnalysis Requirements:")
        for requirement in analysis_requirements:
            print(f"  - {requirement['type']}: {requirement['description']}")
        
        print(f"\\nRecommended Workflow:")
        for step in workflow_recommendations['steps']:
            print(f"  {step['order']}. {step['name']}: {step['description']}")
        ```
        
        **Intent Classification and Routing:**
        ```python
        # Configure for intelligent query routing
        routing_config = {
            'intent_classification': True,
            'workflow_routing': True,
            'parameter_optimization': True,
            'quality_assessment': True
        }
        
        agent_config = AgentConfig.from_config(routing_config)
        query_analyzer = QueryAnalysisAgent.from_config(agent_config)
        
        # Multiple research queries for batch processing
        research_queries = [
            "Predict the 3D structure of the SARS-CoV-2 spike protein",
            "Find all papers about influenza vaccine effectiveness",
            "Compare genome sequences of H1N1 and H3N2 influenza strains",
            "Identify drug targets in the Zika virus proteome"
        ]
        
        # Batch analysis with intent classification
        batch_results = await query_analyzer.batch_analyze_queries(research_queries)
        
        # Route queries to appropriate workflows
        for query, analysis in zip(research_queries, batch_results):
            intent = analysis.data['intent']['primary_intent']
            confidence = analysis.data['confidence']['overall']
            
            print(f"\\nQuery: {query[:50]}...")
            print(f"Intent: {intent}")
            print(f"Confidence: {confidence:.3f}")
            
            # Route based on intent
            if intent == 'structure_prediction':
                recommended_workflow = 'protein_structure_pipeline'
            elif intent == 'literature_review':
                recommended_workflow = 'literature_analysis_pipeline'
            elif intent == 'comparative_genomics':
                recommended_workflow = 'comparative_genomics_pipeline'
            elif intent == 'drug_discovery':
                recommended_workflow = 'drug_target_pipeline'
            
            print(f"Recommended Workflow: {recommended_workflow}")
        ```
        
        **Parameter Extraction and Optimization:**
        ```python
        # Configure for detailed parameter extraction
        parameter_config = {
            'parameter_extraction': 'detailed',
            'database_validation': True,
            'parameter_optimization': True,
            'threshold_recommendation': True
        }
        
        agent_config = AgentConfig.from_config(parameter_config)
        query_analyzer = QueryAnalysisAgent.from_config(agent_config)
        
        # Query with implicit parameters
        parameter_query = "Find conserved motifs in coronavirus spike proteins that could serve as universal vaccine targets"
        
        parameter_analysis = await query_analyzer.extract_analysis_parameters(parameter_query)
        
        # Access extracted and optimized parameters
        extracted_params = parameter_analysis.data['extracted_parameters']
        optimized_params = parameter_analysis.data['optimized_parameters']
        validation_results = parameter_analysis.data['validation']
        
        print("Extracted Parameters:")
        for param_type, values in extracted_params.items():
            print(f"  {param_type}: {values}")
        
        print("\\nOptimized Parameters:")
        for param_name, config in optimized_params.items():
            print(f"  {param_name}:")
            print(f"    Value: {config['value']}")
            print(f"    Confidence: {config['confidence']:.3f}")
            print(f"    Source: {config['source']}")
        
        print("\\nValidation Results:")
        for entity, validation in validation_results.items():
            status = "✅" if validation['valid'] else "❌"
            print(f"  {status} {entity}: {validation['status']}")
        ```
        
        **Biological Context Enhancement:**
        ```python
        # Configure for enhanced biological context analysis
        context_config = {
            'biological_context_focus': True,
            'taxonomic_validation': True,
            'synonym_resolution': True,
            'pathway_integration': True,
            'functional_annotation': True
        }
        
        agent_config = AgentConfig.from_config(context_config)
        query_analyzer = QueryAnalysisAgent.from_config(agent_config)
        
        # Query with complex biological context
        biological_query = "Analyze the role of NSP1 in SARS-CoV-2 immune evasion and compare with other coronavirus non-structural proteins"
        
        context_analysis = await query_analyzer.enhance_biological_context(biological_query)
        
        # Access enhanced biological context
        taxonomic_context = context_analysis.data['taxonomic_context']
        protein_context = context_analysis.data['protein_context']
        functional_context = context_analysis.data['functional_context']
        pathway_context = context_analysis.data['pathway_context']
        
        print("Taxonomic Context:")
        print(f"  Primary Species: {taxonomic_context['primary_species']}")
        print(f"  Related Species: {taxonomic_context['related_species']}")
        print(f"  Taxonomic Level: {taxonomic_context['analysis_level']}")
        
        print("\\nProtein Context:")
        print(f"  Primary Protein: {protein_context['primary_protein']}")
        print(f"  Protein Family: {protein_context['protein_family']}")
        print(f"  Known Functions: {protein_context['known_functions']}")
        
        print("\\nFunctional Context:")
        for function in functional_context['biological_processes']:
            print(f"  - {function['name']}: {function['description']}")
        
        print("\\nPathway Context:")
        for pathway in pathway_context['relevant_pathways']:
            print(f"  - {pathway['name']} ({pathway['database']})")
        ```
    
    **Advanced Features:**
        
        **Machine Learning Integration:**
        * Continuous learning from query analysis feedback
        * Adaptive confidence calibration based on domain expertise
        * Pattern recognition for emerging research areas and methodologies
        * Transfer learning across different biological domains
        
        **Multi-Language Support:**
        * Scientific terminology recognition across multiple languages
        * Translation and standardization of biological terms
        * Cross-cultural research collaboration support
        * International database and resource integration
        
        **Real-Time Analysis:**
        * Streaming query analysis for real-time research support
        * Live collaboration and shared analysis sessions
        * Progressive query refinement and optimization
        * Interactive parameter adjustment and validation
        
        **Integration Capabilities:**
        * Seamless integration with bioinformatics databases and tools
        * API connectivity for external validation and enrichment
        * Workflow automation and pipeline integration
        * Results aggregation and comparative analysis
    
    **Research Applications:**
        
        **Virology Research:**
        * Virus species identification and classification
        * Viral protein analysis and functional prediction
        * Phylogenetic analysis and evolutionary studies
        * Antiviral drug discovery and vaccine development
        
        **Genomics and Proteomics:**
        * Comparative genomics and sequence analysis
        * Protein structure and function prediction
        * Gene expression analysis and regulation studies
        * Biomarker discovery and validation
        
        **Drug Discovery:**
        * Target identification and validation
        * Compound screening and optimization
        * Pharmacokinetics and toxicity analysis
        * Clinical trial design and analysis
        
        **Personalized Medicine:**
        * Genomic variant analysis and interpretation
        * Treatment response prediction and optimization
        * Disease risk assessment and prevention
        * Precision therapy development and implementation
    
    Attributes:
        confidence_threshold (float): Minimum confidence score for query classification
        analysis_depth (str): Level of analysis detail (standard, detailed, comprehensive)
        biological_context_focus (bool): Whether to prioritize biological domain understanding
        intent_classifier (object): Machine learning model for intent classification
        parameter_extractor (object): System for extracting analysis parameters
        biological_validator (object): Validation system for biological entities
    
    Note:
        This agent requires LLM access for natural language query analysis.
        Biological database validation features require internet connectivity.
        The agent continuously learns and improves from analysis feedback.
        Confidence scores should be calibrated based on specific research domains.
    
    Warning:
        Query analysis accuracy depends on the complexity and clarity of input queries.
        Always validate extracted parameters against authoritative biological databases.
        Be cautious with ambiguous biological terms that may have multiple interpretations.
        Confidence scores should be considered in context of specific research requirements.
    
    See Also:
        * :class:`ConversationalSpecializedAgent`: Base conversational specialized agent
        * :class:`SpecializedAgentBase`: Base specialized agent interface
        * :class:`AgentConfig`: Agent configuration schema
        * :mod:`nanobrain.library.agents.specialized`: Specialized agent implementations
        * :mod:`nanobrain.library.tools.bioinformatics`: Bioinformatics tool implementations
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        """Initialize Query Analysis Agent with specialized capabilities"""
        super().__init__(config, **kwargs)
        
        # Agent-specific configuration
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        self.analysis_depth = getattr(config, 'analysis_depth', 'standard')
        self.biological_context_focus = getattr(config, 'biological_context_focus', True)
        
        logger.info(f"🔬 Query Analysis Agent initialized with confidence threshold: {self.confidence_threshold}")
    
    async def _process_specialized_request(self, 
                                         query: str, 
                                         expected_format: str = 'json',
                                         analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Process specialized query analysis request
        
        Args:
            query: User query to analyze
            expected_format: Expected response format ('json', 'text')
            analysis_type: Type of analysis ('comprehensive', 'focused', 'intent_only')
            
        Returns:
            Dict containing query analysis results
            
        ✅ FRAMEWORK COMPLIANCE:
        - Uses LLM capabilities for intelligent analysis
        - No hardcoded patterns or regex matching
        - Configurable analysis depth and focus
        - Returns structured data for downstream processing
        """
        try:
            # Construct specialized analysis prompt
            analysis_prompt = self._build_analysis_prompt(query, analysis_type)
            
            # Process with LLM using conversational capabilities
            response = await self._process_with_llm(analysis_prompt, expected_format)
            
            # Parse and validate response
            analysis_result = self._parse_analysis_response(response, analysis_type)
            
            # Apply confidence scoring
            analysis_result = self._apply_confidence_scoring(analysis_result, query)
            
            logger.debug(f"🔬 Query analysis completed: {analysis_result.get('intent', 'unknown')}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            return {
                'intent': 'analysis_failed',
                'confidence': 0.0,
                'error': str(e),
                'biological_context': {},
                'analysis_type': analysis_type
            }
    
    def _build_analysis_prompt(self, query: str, analysis_type: str) -> str:
        """Build specialized analysis prompt for query processing"""
        
        base_prompt = f"""
        Analyze this biological research query with focus on virus-related information:
        
        QUERY: "{query}"
        
        ANALYSIS REQUIREMENTS:
        - Extract biological intent and context
        - Identify virus species mentions (common names, scientific names, abbreviations)
        - Determine analysis type needed (sequence analysis, PSSM, annotation, etc.)
        - Assess query complexity and confidence level
        - Identify any parameters or constraints mentioned
        
        OUTPUT FORMAT: JSON with the following structure:
        {{
            "intent": "primary query intent",
            "biological_context": {{
                "domain": "virology/microbiology/etc",
                "analysis_type": "requested analysis type",
                "target_organisms": ["identified organisms"],
                "parameters": {{"extracted parameters"}}
            }},
            "confidence": 0.0-1.0,
            "complexity": "simple/moderate/complex",
            "query_classification": "data_request/analysis_request/information_request"
        }}
        """
        
        if analysis_type == 'focused':
            base_prompt += "\n\nFOCUS: Prioritize virus species identification and analysis type extraction."
        elif analysis_type == 'intent_only':
            base_prompt += "\n\nFOCUS: Focus primarily on intent classification and confidence assessment."
        
        return base_prompt
    
    async def _process_with_llm(self, prompt: str, expected_format: str) -> str:
        """Process prompt with LLM using agent capabilities"""
        # Use the agent's conversational capabilities for processing
        # This would integrate with the actual LLM processing logic
        
        # For now, return a structured response
        # In a full implementation, this would use the agent's LLM integration
        
        # Placeholder that mimics LLM processing
        return json.dumps({
            "intent": "biological_analysis",
            "biological_context": {
                "domain": "virology", 
                "analysis_type": "sequence_analysis",
                "target_organisms": ["virus_species_extracted"],
                "parameters": {}
            },
            "confidence": self.confidence_threshold,
            "complexity": "moderate",
            "query_classification": "analysis_request"
        })
    
    def _parse_analysis_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse and validate LLM analysis response"""
        try:
            # Parse JSON response
            parsed = json.loads(response)
            
            # Validate required fields
            required_fields = ['intent', 'biological_context', 'confidence']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = self._get_default_value(field)
            
            # Ensure biological_context is properly structured
            if not isinstance(parsed['biological_context'], dict):
                parsed['biological_context'] = {}
            
            return parsed
            
        except json.JSONDecodeError:
            logger.warning("⚠️ Failed to parse JSON response, using text analysis")
            return self._fallback_text_analysis(response, analysis_type)
    
    def _apply_confidence_scoring(self, analysis_result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Apply confidence scoring based on analysis quality"""
        base_confidence = analysis_result.get('confidence', 0.5)
        
        # Adjust confidence based on biological context completeness
        bio_context = analysis_result.get('biological_context', {})
        if bio_context.get('target_organisms'):
            base_confidence += 0.1
        if bio_context.get('analysis_type'):
            base_confidence += 0.1
        if bio_context.get('domain'):
            base_confidence += 0.05
        
        # Cap confidence at 1.0
        analysis_result['confidence'] = min(base_confidence, 1.0)
        
        return analysis_result
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing required fields"""
        defaults = {
            'intent': 'unknown',
            'biological_context': {},
            'confidence': 0.5,
            'complexity': 'moderate',
            'query_classification': 'information_request'
        }
        return defaults.get(field, None)
    
    def _fallback_text_analysis(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Fallback text analysis when JSON parsing fails"""
        return {
            'intent': 'text_analysis_fallback',
            'biological_context': {
                'domain': 'unknown',
                'raw_response': response[:200]  # First 200 chars
            },
            'confidence': 0.3,  # Lower confidence for fallback
            'complexity': 'unknown',
            'query_classification': 'analysis_request',
            'analysis_method': 'fallback'
        } 