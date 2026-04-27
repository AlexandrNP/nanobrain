#!/usr/bin/env python3
"""
Tool Card Generator for NanoBrain Framework

Automatically generates mandatory Tool Cards for all bioinformatics tools
as required by the A2A protocol.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.card_schemas import (
    ToolCard, IOSchema, ParameterSchema, UsageExample, PerformanceMetrics,
    InputMode, OutputMode, ParameterType, get_card_manager
)


def create_bvbrc_tool_card() -> ToolCard:
    """Create tool card for BV-BRC bioinformatics tool."""
    
    # Input schema
    input_schema = IOSchema(
        format="object",
        description="Parameters for BV-BRC database queries and genomic data retrieval",
        parameters=[
            ParameterSchema(
                name="query_type",
                type=ParameterType.ENUM,
                description="Type of BV-BRC query to perform",
                required=True,
                enum_values=["genome", "protein", "annotation", "feature"],
                example="genome"
            ),
            ParameterSchema(
                name="search_criteria",
                type=ParameterType.OBJECT,
                description="Search criteria for database query",
                required=True,
                example={"organism": "Alphavirus", "genome_status": "Complete"}
            ),
            ParameterSchema(
                name="limit",
                type=ParameterType.INTEGER,
                description="Maximum number of results to retrieve",
                required=False,
                default=100,
                min_value=1,
                max_value=10000,
                example=500
            ),
            ParameterSchema(
                name="fields",
                type=ParameterType.ARRAY,
                description="Specific fields to retrieve from database",
                required=False,
                example=["genome_id", "genome_name", "organism_name", "genome_status"]
            )
        ],
        examples=[
            {
                "query_type": "genome",
                "search_criteria": {"organism": "Alphavirus"},
                "limit": 100,
                "fields": ["genome_id", "genome_name", "organism_name"]
            }
        ],
        content_type="application/json"
    )
    
    # Output schema
    output_schema = IOSchema(
        format="object",
        description="Structured genomic data from BV-BRC database",
        parameters=[
            ParameterSchema(
                name="status",
                type=ParameterType.STRING,
                description="Query execution status",
                required=True,
                enum_values=["success", "error", "partial"],
                example="success"
            ),
            ParameterSchema(
                name="count",
                type=ParameterType.INTEGER,
                description="Number of records retrieved",
                required=True,
                example=45
            ),
            ParameterSchema(
                name="data",
                type=ParameterType.ARRAY,
                description="Array of genomic records",
                required=True,
                example=[{"genome_id": "123", "genome_name": "Example genome"}]
            ),
            ParameterSchema(
                name="metadata",
                type=ParameterType.OBJECT,
                description="Query metadata and execution details",
                required=True,
                example={"execution_time": "2.3s", "cache_hit": False}
            )
        ],
        examples=[
            {
                "status": "success",
                "count": 45,
                "data": [{"genome_id": "123.45", "genome_name": "Alphavirus complete genome"}],
                "metadata": {"execution_time": "2.3s", "cache_hit": False}
            }
        ],
        content_type="application/json"
    )
    
    # Usage examples
    usage_examples = [
        UsageExample(
            name="Alphavirus Genome Search",
            description="Search for complete Alphavirus genomes in BV-BRC database",
            input={
                "query_type": "genome",
                "search_criteria": {"organism": "Alphavirus", "genome_status": "Complete"},
                "limit": 50
            },
            expected_output={
                "status": "success",
                "count": 45,
                "data": [{"genome_id": "123.45", "genome_name": "Alphavirus complete genome"}]
            },
            context="Viral genomics research and comparative analysis",
            tags=["virus", "genome", "alphavirus"]
        ),
        UsageExample(
            name="Protein Feature Query",
            description="Retrieve protein features from specific genome",
            input={
                "query_type": "feature",
                "search_criteria": {"genome_id": "123.45", "feature_type": "CDS"},
                "fields": ["patric_id", "product", "aa_sequence"]
            },
            expected_output={
                "status": "success", 
                "count": 12,
                "data": [{"patric_id": "fig|123.45.peg.1", "product": "viral protein"}]
            },
            context="Protein annotation and functional analysis",
            tags=["protein", "annotation", "features"]
        )
    ]
    
    # Performance metrics
    performance = PerformanceMetrics(
        typical_response_time="2-10 seconds",
        memory_usage="50-200 MB",
        cpu_requirements="Low",
        concurrency_limit=5,
        rate_limit="100 requests/hour",
        scaling_characteristics="Good for batch processing"
    )
    
    return ToolCard(
        name="BVBRCTool",
        version="1.0.0",
        description="Bioinformatics Viral Bacterial Resource Center (BV-BRC) database access tool",
        purpose="Provides programmatic access to BV-BRC database for retrieving viral and bacterial genomic data, annotations, and metadata. Enables comprehensive querying of genomes, proteins, features, and functional annotations.",
        category="bioinformatics",
        subcategory="database_access",
        tags=["viral_genomics", "bacterial_genomics", "database", "annotation", "bvbrc", "patric"],
        input_modes=[InputMode.JSON, InputMode.DATA],
        output_modes=[OutputMode.JSON, OutputMode.STRUCTURED],
        input_schema=input_schema,
        output_schema=output_schema,
        usage_examples=usage_examples,
        common_use_cases=[
            "Viral genome database queries",
            "Bacterial pathogen research",
            "Comparative genomics studies",
            "Protein annotation retrieval",
            "Functional genomics analysis"
        ],
        limitations=[
            "Requires internet connection for database access",
            "Rate limited by BV-BRC server policies",
            "Large queries may timeout",
            "Dependent on BV-BRC database availability"
        ],
        dependencies=["requests", "pandas", "biopython"],
        system_requirements=["Python 3.8+", "Internet connection"],
        supported_formats=["JSON", "CSV", "FASTA"],
        performance=performance,
        a2a_compatible=True,
        a2a_version="1.0.0",
        authentication_required=False,
        authentication_schemes=[],
        provider="NanoBrain Bioinformatics",
        license="MIT",
        documentation_url="https://nanobrain.readthedocs.io/tools/bvbrc",
        source_url="https://github.com/nanobrain/tools/bvbrc",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat()
    )


def create_mmseqs2_tool_card() -> ToolCard:
    """Create tool card for MMseqs2 sequence alignment tool."""
    
    # Input schema
    input_schema = IOSchema(
        format="object",
        description="Parameters for MMseqs2 sequence clustering and alignment",
        parameters=[
            ParameterSchema(
                name="sequences",
                type=ParameterType.ARRAY,
                description="Array of protein sequences to cluster",
                required=True,
                example=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"]
            ),
            ParameterSchema(
                name="cluster_mode",
                type=ParameterType.ENUM,
                description="Clustering algorithm mode",
                required=False,
                default="cluster",
                enum_values=["cluster", "linclust", "cascaded"],
                example="cluster"
            ),
            ParameterSchema(
                name="min_seq_id",
                type=ParameterType.FLOAT,
                description="Minimum sequence identity threshold",
                required=False,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                example=0.5
            ),
            ParameterSchema(
                name="coverage",
                type=ParameterType.FLOAT,
                description="Minimum coverage threshold",
                required=False,
                default=0.8,
                min_value=0.0,
                max_value=1.0,
                example=0.8
            ),
            ParameterSchema(
                name="sensitivity",
                type=ParameterType.FLOAT,
                description="Search sensitivity parameter",
                required=False,
                default=7.5,
                min_value=1.0,
                max_value=8.0,
                example=7.5
            )
        ],
        examples=[
            {
                "sequences": ["MKTAYIAK...", "MRVLQIAK..."],
                "cluster_mode": "cluster",
                "min_seq_id": 0.5,
                "coverage": 0.8
            }
        ],
        content_type="application/json"
    )
    
    # Output schema
    output_schema = IOSchema(
        format="object",
        description="MMseqs2 clustering results with cluster assignments",
        parameters=[
            ParameterSchema(
                name="clusters",
                type=ParameterType.ARRAY,
                description="Array of protein clusters",
                required=True,
                example=[{"cluster_id": 0, "representative": "seq_1", "members": ["seq_1", "seq_2"]}]
            ),
            ParameterSchema(
                name="statistics",
                type=ParameterType.OBJECT,
                description="Clustering statistics",
                required=True,
                example={"total_sequences": 100, "total_clusters": 25, "singleton_clusters": 5}
            ),
            ParameterSchema(
                name="execution_time",
                type=ParameterType.FLOAT,
                description="Execution time in seconds",
                required=True,
                example=45.2
            )
        ],
        examples=[
            {
                "clusters": [{"cluster_id": 0, "representative": "seq_1", "members": ["seq_1", "seq_2"]}],
                "statistics": {"total_sequences": 100, "total_clusters": 25},
                "execution_time": 45.2
            }
        ],
        content_type="application/json"
    )
    
    # Usage examples
    usage_examples = [
        UsageExample(
            name="Protein Clustering",
            description="Cluster viral proteins by sequence similarity",
            input={
                "sequences": ["MKTAYIAKQRQISFVK...", "MRVLQIAKQRQISFVK..."],
                "cluster_mode": "cluster",
                "min_seq_id": 0.5,
                "coverage": 0.8
            },
            expected_output={
                "clusters": [{"cluster_id": 0, "representative": "seq_1", "members": ["seq_1", "seq_2"]}],
                "statistics": {"total_sequences": 2, "total_clusters": 1}
            },
            context="Viral protein family analysis",
            tags=["clustering", "viral_proteins", "similarity"]
        )
    ]
    
    # Performance metrics
    performance = PerformanceMetrics(
        typical_response_time="30 seconds - 5 minutes",
        memory_usage="1-10 GB",
        cpu_requirements="High (multi-core recommended)",
        concurrency_limit=2,
        rate_limit="10 jobs/hour",
        scaling_characteristics="Excellent for large datasets"
    )
    
    return ToolCard(
        name="MMseqs2Tool",
        version="1.0.0",
        description="Ultra-fast protein sequence clustering and search tool",
        purpose="Performs high-speed protein sequence clustering, similarity search, and database creation using MMseqs2 algorithm. Optimized for large-scale protein analysis and functional annotation.",
        category="bioinformatics",
        subcategory="sequence_analysis",
        tags=["clustering", "sequence_alignment", "protein_analysis", "similarity_search", "mmseqs2"],
        input_modes=[InputMode.DATA, InputMode.FILE],
        output_modes=[OutputMode.STRUCTURED, OutputMode.FILE],
        input_schema=input_schema,
        output_schema=output_schema,
        usage_examples=usage_examples,
        common_use_cases=[
            "Protein sequence clustering",
            "Homology detection",
            "Database creation",
            "Functional annotation",
            "Protein family analysis"
        ],
        limitations=[
            "High memory requirements for large datasets",
            "CPU intensive operations",
            "Requires local MMseqs2 installation",
            "Limited to protein sequences"
        ],
        dependencies=["mmseqs2", "biopython", "subprocess"],
        system_requirements=["MMseqs2 binary", "8GB+ RAM", "Multi-core CPU"],
        supported_formats=["FASTA", "JSON"],
        performance=performance,
        a2a_compatible=True,
        a2a_version="1.0.0",
        authentication_required=False,
        authentication_schemes=[],
        provider="NanoBrain Bioinformatics",
        license="MIT",
        documentation_url="https://nanobrain.readthedocs.io/tools/mmseqs2",
        source_url="https://github.com/nanobrain/tools/mmseqs2",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat()
    )


def create_muscle_tool_card() -> ToolCard:
    """Create tool card for MUSCLE multiple sequence alignment tool."""
    
    # Input schema
    input_schema = IOSchema(
        format="object",
        description="Parameters for MUSCLE multiple sequence alignment",
        parameters=[
            ParameterSchema(
                name="sequences",
                type=ParameterType.ARRAY,
                description="Array of sequences to align",
                required=True,
                example=["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK"]
            ),
            ParameterSchema(
                name="sequence_type",
                type=ParameterType.ENUM,
                description="Type of sequences being aligned",
                required=False,
                default="protein",
                enum_values=["protein", "dna", "rna"],
                example="protein"
            ),
            ParameterSchema(
                name="max_iterations",
                type=ParameterType.INTEGER,
                description="Maximum number of refinement iterations",
                required=False,
                default=16,
                min_value=1,
                max_value=100,
                example=16
            ),
            ParameterSchema(
                name="diagonal_optimization",
                type=ParameterType.BOOLEAN,
                description="Enable diagonal optimization for speed",
                required=False,
                default=True,
                example=True
            )
        ],
        examples=[
            {
                "sequences": ["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK"],
                "sequence_type": "protein",
                "max_iterations": 16
            }
        ],
        content_type="application/json"
    )
    
    # Output schema
    output_schema = IOSchema(
        format="object",
        description="Multiple sequence alignment results",
        parameters=[
            ParameterSchema(
                name="alignment",
                type=ParameterType.ARRAY,
                description="Aligned sequences with gaps",
                required=True,
                example=["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK"]
            ),
            ParameterSchema(
                name="alignment_score",
                type=ParameterType.FLOAT,
                description="Overall alignment quality score",
                required=True,
                example=0.85
            ),
            ParameterSchema(
                name="conservation_profile",
                type=ParameterType.ARRAY,
                description="Per-position conservation scores",
                required=True,
                example=[0.95, 0.80, 0.90, 0.75]
            ),
            ParameterSchema(
                name="execution_time",
                type=ParameterType.FLOAT,
                description="Alignment execution time in seconds",
                required=True,
                example=12.5
            )
        ],
        examples=[
            {
                "alignment": ["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK"],
                "alignment_score": 0.85,
                "conservation_profile": [0.95, 0.80, 0.90, 0.75],
                "execution_time": 12.5
            }
        ],
        content_type="application/json"
    )
    
    # Usage examples
    usage_examples = [
        UsageExample(
            name="Viral Protein Alignment",
            description="Align viral protein sequences to identify conserved regions",
            input={
                "sequences": ["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK", "MKTAYIAKQRQISGVK"],
                "sequence_type": "protein",
                "max_iterations": 16
            },
            expected_output={
                "alignment": ["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK", "MKTAYIAKQRQISGVK"],
                "alignment_score": 0.85,
                "conservation_profile": [0.95, 0.80, 0.90, 0.75]
            },
            context="Viral protein family analysis and evolution",
            tags=["alignment", "viral_proteins", "conservation"]
        )
    ]
    
    # Performance metrics  
    performance = PerformanceMetrics(
        typical_response_time="10-60 seconds",
        memory_usage="100 MB - 2 GB",
        cpu_requirements="Medium",
        concurrency_limit=3,
        rate_limit="20 alignments/hour",
        scaling_characteristics="Good for medium-sized datasets"
    )
    
    return ToolCard(
        name="MUSCLETool",
        version="1.0.0", 
        description="Multiple sequence alignment tool using MUSCLE algorithm",
        purpose="Performs high-quality multiple sequence alignment of protein, DNA, or RNA sequences using the MUSCLE algorithm. Provides alignment quality scoring and conservation analysis.",
        category="bioinformatics",
        subcategory="sequence_alignment",
        tags=["multiple_alignment", "muscle", "conservation", "phylogeny", "evolution"],
        input_modes=[InputMode.DATA, InputMode.FILE],
        output_modes=[OutputMode.STRUCTURED, OutputMode.FILE],
        input_schema=input_schema,
        output_schema=output_schema,
        usage_examples=usage_examples,
        common_use_cases=[
            "Multiple sequence alignment",
            "Phylogenetic analysis preparation",
            "Conservation analysis",
            "Protein family studies",
            "Evolutionary analysis"
        ],
        limitations=[
            "Performance degrades with very large alignments",
            "Memory intensive for long sequences",
            "Requires local MUSCLE installation",
            "Limited support for structural information"
        ],
        dependencies=["muscle", "biopython"],
        system_requirements=["MUSCLE binary", "2GB+ RAM", "Unix-like system"],
        supported_formats=["FASTA", "CLUSTAL", "JSON"],
        performance=performance,
        a2a_compatible=True,
        a2a_version="1.0.0",
        authentication_required=False,
        authentication_schemes=[],
        provider="NanoBrain Bioinformatics",
        license="MIT",
        documentation_url="https://nanobrain.readthedocs.io/tools/muscle",
        source_url="https://github.com/nanobrain/tools/muscle",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat()
    )


def create_pubmed_client_card() -> ToolCard:
    """Create tool card for PubMed research literature client."""
    
    # Input schema
    input_schema = IOSchema(
        format="object",
        description="Parameters for PubMed literature search and retrieval",
        parameters=[
            ParameterSchema(
                name="query",
                type=ParameterType.STRING,
                description="PubMed search query string",
                required=True,
                example="viral protein structure alphavirus"
            ),
            ParameterSchema(
                name="max_results",
                type=ParameterType.INTEGER,
                description="Maximum number of articles to retrieve",
                required=False,
                default=20,
                min_value=1,
                max_value=1000,
                example=50
            ),
            ParameterSchema(
                name="publication_date",
                type=ParameterType.STRING,
                description="Publication date filter (YYYY/MM/DD or date range)",
                required=False,
                pattern=r"^\d{4}(/\d{2}(/\d{2})?)?(\:\d{4}(/\d{2}(/\d{2})?)?)?$",
                example="2020:2024"
            ),
            ParameterSchema(
                name="fields",
                type=ParameterType.ARRAY,
                description="Specific fields to retrieve",
                required=False,
                default=["title", "abstract", "authors", "doi"],
                example=["title", "abstract", "authors", "doi", "journal"]
            )
        ],
        examples=[
            {
                "query": "viral protein structure alphavirus",
                "max_results": 50,
                "publication_date": "2020:2024",
                "fields": ["title", "abstract", "authors", "doi"]
            }
        ],
        content_type="application/json"
    )
    
    # Output schema
    output_schema = IOSchema(
        format="object",
        description="PubMed search results with publication metadata",
        parameters=[
            ParameterSchema(
                name="total_found",
                type=ParameterType.INTEGER,
                description="Total number of matching articles found",
                required=True,
                example=1247
            ),
            ParameterSchema(
                name="articles",
                type=ParameterType.ARRAY,
                description="Array of article metadata",
                required=True,
                example=[{"pmid": "12345678", "title": "Viral protein analysis"}]
            ),
            ParameterSchema(
                name="search_metadata",
                type=ParameterType.OBJECT,
                description="Search execution metadata",
                required=True,
                example={"execution_time": "1.2s", "query_translation": "viral[All Fields] AND protein[All Fields]"}
            )
        ],
        examples=[
            {
                "total_found": 1247,
                "articles": [{"pmid": "12345678", "title": "Viral protein structure analysis"}],
                "search_metadata": {"execution_time": "1.2s"}
            }
        ],
        content_type="application/json"
    )
    
    # Usage examples
    usage_examples = [
        UsageExample(
            name="Viral Research Literature Search",
            description="Search for recent publications on viral protein structures",
            input={
                "query": "viral protein structure alphavirus",
                "max_results": 20,
                "publication_date": "2020:2024"
            },
            expected_output={
                "total_found": 156,
                "articles": [{"pmid": "12345678", "title": "Alphavirus protein structure and function"}],
                "search_metadata": {"execution_time": "1.2s"}
            },
            context="Literature review for viral research project",
            tags=["literature", "viral_research", "proteins"]
        )
    ]
    
    # Performance metrics
    performance = PerformanceMetrics(
        typical_response_time="1-5 seconds",
        memory_usage="10-50 MB",
        cpu_requirements="Low",
        concurrency_limit=10,
        rate_limit="300 queries/hour",
        scaling_characteristics="Good for batch literature processing"
    )
    
    return ToolCard(
        name="PubMedClient",
        version="1.0.0",
        description="Scientific literature search and retrieval client for PubMed database",
        purpose="Provides programmatic access to PubMed database for searching and retrieving scientific publications. Enables literature mining, bibliometric analysis, and research context gathering.",
        category="bioinformatics",
        subcategory="literature_mining",
        tags=["pubmed", "literature", "research", "publications", "bibliography", "scientific_search"],
        input_modes=[InputMode.TEXT, InputMode.JSON],
        output_modes=[OutputMode.JSON, OutputMode.STRUCTURED],
        input_schema=input_schema,
        output_schema=output_schema,
        usage_examples=usage_examples,
        common_use_cases=[
            "Literature review automation",
            "Research trend analysis",
            "Citation network analysis",
            "Scientific knowledge discovery",
            "Publication metadata extraction"
        ],
        limitations=[
            "Rate limited by NCBI policies",
            "Requires internet connection",
            "Abstract text may be truncated",
            "Full text access depends on publisher"
        ],
        dependencies=["biopython", "requests", "xml.etree"],
        system_requirements=["Python 3.8+", "Internet connection"],
        supported_formats=["JSON", "XML", "CSV"],
        performance=performance,
        a2a_compatible=True,
        a2a_version="1.0.0",
        authentication_required=False,
        authentication_schemes=["api_key_optional"],
        provider="NanoBrain Bioinformatics",
        license="MIT",
        documentation_url="https://nanobrain.readthedocs.io/tools/pubmed",
        source_url="https://github.com/nanobrain/tools/pubmed",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat()
    )


def main():
    """Generate all tool cards for NanoBrain bioinformatics tools."""
    print("🔧 Generating Tool Cards for NanoBrain Framework...")
    
    # Initialize card manager
    card_manager = get_card_manager("cards")
    
    # Generate tool cards
    tools = [
        ("BVBRCTool", create_bvbrc_tool_card),
        ("MMseqs2Tool", create_mmseqs2_tool_card),
        ("MUSCLETool", create_muscle_tool_card),
        ("PubMedClient", create_pubmed_client_card)
    ]
    
    generated_cards = []
    
    for tool_name, card_creator in tools:
        print(f"  📋 Generating {tool_name} card...")
        try:
            tool_card = card_creator()
            card_manager.register_tool_card(tool_card)
            
            # Save in both YAML and JSON formats
            yaml_path = card_manager.save_tool_card(tool_card, "yaml")
            json_path = card_manager.save_tool_card(tool_card, "json")
            
            generated_cards.append({
                "name": tool_name,
                "yaml_path": yaml_path,
                "json_path": json_path
            })
            
            print(f"    ✅ {tool_name} card generated successfully")
            print(f"       YAML: {yaml_path}")
            print(f"       JSON: {json_path}")
            
        except Exception as e:
            print(f"    ❌ Failed to generate {tool_name} card: {e}")
    
    print(f"\n🎉 Tool Card Generation Complete!")
    print(f"Generated {len(generated_cards)} tool cards:")
    
    for card in generated_cards:
        print(f"  • {card['name']}")
    
    print(f"\nCards saved to: {card_manager.cards_directory}")
    print("Cards are A2A protocol compliant and ready for agent discovery!")


if __name__ == "__main__":
    main()