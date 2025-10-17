#!/usr/bin/env python3
"""
Stress Test: 1000 Concurrent Queries
=====================================

Comprehensive stress test for parallel RAG query journey tracking system.
Processes 1000 realistic biological queries with complete tracking.
"""

import asyncio
import sys
import time
import random
import psutil
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.parallel_rag_with_parsl.journey_logging.journey_logger import QueryJourneyLogger
from demos.parallel_rag_with_parsl.models.query_journey import Document
from demos.parallel_rag_with_parsl.tools.analyze_logs import LogAnalyzer
from demos.parallel_rag_with_parsl.tools.export_logs import LogExporter


# Query templates for generating 1000 diverse queries
QUERY_TEMPLATES = {
    'viral_mechanisms': [
        "What are the key mechanisms of {virus} {process}?",
        "How does {virus} achieve {process}?",
        "Explain the molecular basis of {virus} {process}.",
        "What are the structural features enabling {virus} {process}?",
        "Describe the {process} pathway in {virus} infection.",
    ],
    'protein_function': [
        "How does the {protein} facilitate {function}?",
        "What is the role of {protein} in {function}?",
        "Explain the mechanism of {protein}-mediated {function}.",
        "What are the structural domains of {protein} involved in {function}?",
        "How is {protein} regulated during {function}?",
    ],
    'molecular_interactions': [
        "What are the key interactions between {molecule1} and {molecule2}?",
        "How does {molecule1} binding to {molecule2} affect {process}?",
        "Explain the molecular basis of {molecule1}-{molecule2} interaction.",
        "What residues are critical for {molecule1}-{molecule2} binding?",
        "How does the {molecule1}-{molecule2} complex regulate {process}?",
    ],
    'therapeutic_targets': [
        "What are potential therapeutic targets for blocking {process}?",
        "How can {protein} be targeted for antiviral therapy?",
        "What inhibitors are effective against {protein}?",
        "Explain the rationale for targeting {protein} in {disease}.",
        "What are the challenges in developing {protein} inhibitors?",
    ],
    'structural_biology': [
        "What is the structure of {protein} in complex with {molecule}?",
        "How does {protein} undergo conformational changes during {process}?",
        "What are the key structural features of {protein}?",
        "Explain the structural basis of {protein} function.",
        "How does {mutation} affect {protein} structure and function?",
    ]
}

# Vocabulary for generating diverse queries
VIRUSES = [
    "SARS-CoV-2", "influenza virus", "HIV", "hepatitis C virus", "dengue virus",
    "Zika virus", "Ebola virus", "measles virus", "herpes simplex virus",
    "respiratory syncytial virus", "adenovirus", "rotavirus", "norovirus",
    "poliovirus", "rabies virus", "West Nile virus", "chikungunya virus"
]

PROTEINS = [
    "spike protein", "hemagglutinin", "neuraminidase", "envelope protein",
    "capsid protein", "polymerase", "protease", "reverse transcriptase",
    "integrase", "fusion protein", "nucleoprotein", "matrix protein",
    "glycoprotein", "phosphoprotein", "NS1 protein", "VP1 protein"
]

PROCESSES = [
    "membrane fusion", "viral entry", "genome replication", "assembly",
    "budding", "receptor binding", "endocytosis", "uncoating",
    "nuclear import", "transcription", "translation", "maturation",
    "immune evasion", "cell tropism", "pathogenesis"
]

FUNCTIONS = [
    "viral entry", "membrane fusion", "receptor binding", "proteolytic cleavage",
    "genome packaging", "viral assembly", "immune evasion", "host cell attachment",
    "endosomal escape", "nuclear localization", "RNA synthesis", "protein processing"
]

MOLECULES = [
    "ACE2", "sialic acid", "heparan sulfate", "CD4", "CXCR4", "integrin",
    "transferrin receptor", "claudin", "occludin", "JAM-A", "nectin",
    "ICAM-1", "VCAM-1", "DC-SIGN", "L-SIGN", "TIM-1", "TAM receptors"
]

DISEASES = [
    "COVID-19", "influenza", "AIDS", "hepatitis", "dengue fever",
    "Zika infection", "Ebola disease", "measles", "herpes infection",
    "RSV infection", "adenovirus infection", "rotavirus gastroenteritis"
]

MUTATIONS = [
    "D614G", "N501Y", "E484K", "K417N", "L452R", "T478K", "P681R",
    "H655Y", "N679K", "P681H", "A570D", "S982A", "D1118H"
]


def generate_queries(n: int = 1000) -> list:
    """Generate N diverse biological queries."""
    queries = []
    
    for i in range(n):
        # Select random template category
        category = random.choice(list(QUERY_TEMPLATES.keys()))
        template = random.choice(QUERY_TEMPLATES[category])
        
        # Fill template with random vocabulary
        query = template
        if '{virus}' in query:
            query = query.replace('{virus}', random.choice(VIRUSES))
        if '{protein}' in query:
            query = query.replace('{protein}', random.choice(PROTEINS))
        if '{process}' in query:
            query = query.replace('{process}', random.choice(PROCESSES))
        if '{function}' in query:
            query = query.replace('{function}', random.choice(FUNCTIONS))
        if '{molecule1}' in query:
            query = query.replace('{molecule1}', random.choice(MOLECULES))
        if '{molecule2}' in query:
            query = query.replace('{molecule2}', random.choice(MOLECULES))
        if '{molecule}' in query:
            query = query.replace('{molecule}', random.choice(MOLECULES))
        if '{disease}' in query:
            query = query.replace('{disease}', random.choice(DISEASES))
        if '{mutation}' in query:
            query = query.replace('{mutation}', random.choice(MUTATIONS))
        
        queries.append(query)
    
    return queries


def generate_realistic_documents(query: str, num_docs: int = 3) -> list:
    """Generate realistic documents based on query."""
    documents = []
    
    # Extract key terms from query
    query_lower = query.lower()
    
    for i in range(num_docs):
        # Generate realistic content based on query
        content_templates = [
            f"Recent studies on {query[:50]}... demonstrate critical molecular mechanisms involving structural rearrangements and protein-protein interactions. The process requires coordinated conformational changes and is regulated by multiple cellular factors.",
            f"Research investigating {query[:50]}... reveals important insights into the molecular basis of viral infection. Key findings include the identification of critical binding sites and the characterization of intermediate states.",
            f"Analysis of {query[:50]}... shows that the mechanism involves multiple sequential steps, each representing a potential therapeutic target. Understanding these processes is crucial for developing effective interventions.",
        ]
        
        content = random.choice(content_templates)
        
        # Generate realistic metadata
        doc_id = f"PMC{7000000 + random.randint(0, 999999)}"
        year = random.randint(2020, 2024)
        relevance = 0.95 - (i * 0.03) + random.uniform(-0.02, 0.02)
        
        documents.append(Document(
            doc_id=doc_id,
            content=content,
            relevance_score=relevance,
            metadata={
                'source': 'PubMed Central',
                'year': year,
                'pmid': doc_id,
                'title': f"Study on {query[:60]}..."
            }
        ))
    
    return documents


async def process_query(logger, query_id, query, worker_id, semaphore):
    """Process a single query through the workflow."""
    async with semaphore:  # Limit concurrent processing
        try:
            # Start journey
            logger.start_journey(query_id, query)
            
            # Step 1: Query Enhancement
            step1_start = time.time()
            await asyncio.sleep(random.uniform(0.05, 0.15))  # Simulate processing
            step1_end = time.time()
            
            enhanced_query = f"Provide a comprehensive analysis of {query.lower()} Include molecular mechanisms, structural biology, and clinical implications."
            
            logger.update_enhancement(
                query_id=query_id,
                enhanced_query=enhanced_query,
                worker_id=worker_id,
                instance_id=hash(query_id),
                start_time=step1_start,
                end_time=step1_end,
                duration=step1_end - step1_start
            )
            
            # Step 2: Document Retrieval
            step2_start = time.time()
            await asyncio.sleep(random.uniform(0.10, 0.20))  # Simulate processing
            step2_end = time.time()
            
            documents = generate_realistic_documents(query)
            
            logger.update_retrieval(
                query_id=query_id,
                documents=documents,
                worker_id=worker_id,
                instance_id=hash(query_id) + 1000,
                start_time=step2_start,
                end_time=step2_end,
                duration=step2_end - step2_start
            )
            
            # Step 3: Response Generation
            step3_start = time.time()
            await asyncio.sleep(random.uniform(0.15, 0.25))  # Simulate processing
            step3_end = time.time()
            
            final_response = f"""Based on the comprehensive analysis of the retrieved literature, here are the key findings regarding {query}:

**Molecular Mechanisms:**
{documents[0].content[:150]}...

**Structural Biology:**
{documents[1].content[:150] if len(documents) > 1 else 'Additional structural details...'}...

**Clinical Implications:**
The understanding of these mechanisms provides important insights for therapeutic development and intervention strategies.

**Conclusion:**
These findings highlight the complex molecular processes involved and suggest multiple points for therapeutic intervention.
"""
            
            logger.update_generation(
                query_id=query_id,
                final_response=final_response,
                worker_id=worker_id,
                instance_id=hash(query_id) + 2000,
                start_time=step3_start,
                end_time=step3_end,
                duration=step3_end - step3_start
            )
            
            # Complete journey
            logger.complete_journey(query_id)
            logger.save_journey(query_id)
            
            return True
            
        except Exception as e:
            logger.fail_journey(query_id, str(e))
            logger.save_journey(query_id)
            return False


async def main():
    """Run stress test."""
    print("="*80)
    print("STRESS TEST: 1000 CONCURRENT QUERIES")
    print("="*80)
    print(f"\nTest started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    num_queries = 1000
    num_workers = 4
    max_concurrent = 50  # Limit concurrent tasks to avoid overwhelming system
    
    print(f"\nConfiguration:")
    print(f"  Total queries: {num_queries}")
    print(f"  Workers: {num_workers}")
    print(f"  Max concurrent: {max_concurrent}")
    
    # Setup
    output_dir = "demos/parallel_rag_with_parsl/output/stress_test_1000"
    logger = QueryJourneyLogger(output_dir=output_dir, format="both")
    
    print(f"\n✓ Created journey logger: {output_dir}")
    
    # Generate queries
    print(f"\n{'─'*80}")
    print("Generating Queries")
    print(f"{'─'*80}")
    
    queries = generate_queries(num_queries)
    print(f"✓ Generated {len(queries)} diverse biological queries")
    
    # Show sample queries
    print(f"\nSample queries:")
    for i in range(min(5, len(queries))):
        print(f"  {i+1}. {queries[i]}")
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\n{'─'*80}")
    print("Processing Queries")
    print(f"{'─'*80}")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Process queries
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    for i, query in enumerate(queries):
        query_id = f"q_stress_{i+1:04d}"
        worker_id = f"worker_{i % num_workers}"
        
        task = process_query(logger, query_id, query, worker_id, semaphore)
        tasks.append(task)
    
    # Process with progress updates
    completed = 0
    failed = 0
    
    for i, task in enumerate(asyncio.as_completed(tasks)):
        result = await task
        if result:
            completed += 1
        else:
            failed += 1
        
        # Progress update every 50 queries
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_queries - (i + 1)) / rate if rate > 0 else 0
            current_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"Progress: {i+1}/{num_queries} ({(i+1)/num_queries*100:.1f}%) | "
                  f"Rate: {rate:.1f} q/s | ETA: {eta:.0f}s | "
                  f"Memory: {current_memory:.2f} MB")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    print(f"\n✓ All queries processed!")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Final memory: {final_memory:.2f} MB (+{memory_increase:.2f} MB)")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS AND RESULTS")
    print(f"{'='*80}")

    analyzer = LogAnalyzer(output_dir)

    # Performance metrics
    print(f"\n{'─'*80}")
    print("Performance Metrics")
    print(f"{'─'*80}")

    stats = analyzer.get_performance_stats()

    print(f"\nThroughput:")
    print(f"  Queries per second: {num_queries / total_time:.2f}")
    print(f"  Average time per query: {total_time / num_queries:.3f}s")

    print(f"\nStep Performance:")
    print(f"  Avg enhancement time: {stats['avg_enhancement_time']:.3f}s")
    print(f"  Avg retrieval time: {stats['avg_retrieval_time']:.3f}s")
    print(f"  Avg generation time: {stats['avg_generation_time']:.3f}s")
    print(f"  Avg total time: {stats['avg_total_time']:.3f}s")

    print(f"\nTime Range:")
    print(f"  Min total time: {stats['min_total_time']:.3f}s")
    print(f"  Max total time: {stats['max_total_time']:.3f}s")

    print(f"\nQuality:")
    print(f"  Avg relevance score: {stats['avg_relevance_score']:.2f}")

    print(f"\nStatus:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Complete: {stats['complete']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Pending: {stats['pending']}")

    # Worker utilization
    print(f"\n{'─'*80}")
    print("Worker Utilization")
    print(f"{'─'*80}")

    worker_stats = analyzer.get_worker_stats()
    total_tasks = sum(worker_stats.values())

    for worker_id in sorted(worker_stats.keys()):
        count = worker_stats[worker_id]
        percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
        print(f"  {worker_id}: {count} tasks ({percentage:.1f}%)")

    # Slowest queries
    print(f"\n{'─'*80}")
    print("Top 10 Slowest Queries")
    print(f"{'─'*80}")

    slowest = analyzer.get_slowest_queries(10)
    for i, stat in enumerate(slowest, 1):
        journey = analyzer.journeys[stat.query_id]
        print(f"  {i}. {stat.query_id}: {stat.total_time:.3f}s")
        print(f"     Query: {journey.original_query[:60]}...")

    # Lowest relevance
    print(f"\n{'─'*80}")
    print("Top 10 Lowest Relevance Queries")
    print(f"{'─'*80}")

    lowest = analyzer.get_lowest_relevance_queries(10)
    for i, stat in enumerate(lowest, 1):
        journey = analyzer.journeys[stat.query_id]
        print(f"  {i}. {stat.query_id}: {stat.avg_relevance:.2f}")
        print(f"     Query: {journey.original_query[:60]}...")

    # Memory analysis
    print(f"\n{'─'*80}")
    print("Memory Analysis")
    print(f"{'─'*80}")

    print(f"  Initial memory: {initial_memory:.2f} MB")
    print(f"  Final memory: {final_memory:.2f} MB")
    print(f"  Memory increase: {memory_increase:.2f} MB")
    print(f"  Memory per query: {memory_increase / num_queries:.3f} MB")

    # Tracking overhead
    print(f"\n{'─'*80}")
    print("Tracking Overhead Analysis")
    print(f"{'─'*80}")

    # Estimate overhead (tracking adds ~5-10ms per query)
    estimated_processing_time = stats['avg_enhancement_time'] + stats['avg_retrieval_time'] + stats['avg_generation_time']
    estimated_overhead = stats['avg_total_time'] - estimated_processing_time
    overhead_percentage = (estimated_overhead / stats['avg_total_time'] * 100) if stats['avg_total_time'] > 0 else 0

    print(f"  Estimated processing time: {estimated_processing_time:.3f}s")
    print(f"  Estimated tracking overhead: {estimated_overhead:.3f}s")
    print(f"  Overhead percentage: {overhead_percentage:.1f}%")

    # Export results
    print(f"\n{'─'*80}")
    print("Exporting Results")
    print(f"{'─'*80}")

    export_dir = Path(output_dir) / "exports"
    export_dir.mkdir(exist_ok=True)

    exporter = LogExporter(analyzer)

    print(f"\nExporting to {export_dir}/")
    exporter.export_to_csv(str(export_dir / "all_queries.csv"))
    exporter.export_summary_to_text(str(export_dir / "summary.txt"))
    exporter.export_detailed_report(str(export_dir / "detailed_report.txt"))

    # Export slowest queries
    slowest_ids = [stat.query_id for stat in slowest]
    exporter.export_to_csv(str(export_dir / "slowest_queries.csv"), queries=slowest_ids)

    # Export lowest relevance queries
    lowest_ids = [stat.query_id for stat in lowest]
    exporter.export_to_csv(str(export_dir / "lowest_relevance_queries.csv"), queries=lowest_ids)

    # Create performance report
    report_file = export_dir / "performance_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRESS TEST PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {num_queries}\n")
        f.write(f"Workers: {num_workers}\n")
        f.write(f"Max Concurrent: {max_concurrent}\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total execution time: {total_time:.2f}s\n")
        f.write(f"Throughput: {num_queries / total_time:.2f} queries/second\n")
        f.write(f"Average time per query: {total_time / num_queries:.3f}s\n\n")

        f.write("STEP PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Avg enhancement time: {stats['avg_enhancement_time']:.3f}s\n")
        f.write(f"Avg retrieval time: {stats['avg_retrieval_time']:.3f}s\n")
        f.write(f"Avg generation time: {stats['avg_generation_time']:.3f}s\n")
        f.write(f"Avg total time: {stats['avg_total_time']:.3f}s\n\n")

        f.write("QUALITY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Avg relevance score: {stats['avg_relevance_score']:.2f}\n\n")

        f.write("MEMORY USAGE\n")
        f.write("-"*80 + "\n")
        f.write(f"Initial memory: {initial_memory:.2f} MB\n")
        f.write(f"Final memory: {final_memory:.2f} MB\n")
        f.write(f"Memory increase: {memory_increase:.2f} MB\n")
        f.write(f"Memory per query: {memory_increase / num_queries:.3f} MB\n\n")

        f.write("TRACKING OVERHEAD\n")
        f.write("-"*80 + "\n")
        f.write(f"Estimated overhead: {estimated_overhead:.3f}s ({overhead_percentage:.1f}%)\n\n")

        f.write("STATUS\n")
        f.write("-"*80 + "\n")
        f.write(f"Completed: {stats['complete']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Success rate: {stats['complete'] / stats['total_queries'] * 100:.1f}%\n")

    print(f"✓ Performance report saved to {report_file}")

    # Verify files
    print(f"\n{'─'*80}")
    print("Verification")
    print(f"{'─'*80}")

    queries_dir = Path(output_dir) / "queries"
    json_files = list(queries_dir.glob("*.json"))
    text_files = list(queries_dir.glob("*.txt"))

    print(f"\nLog files:")
    print(f"  JSON files: {len(json_files)}")
    print(f"  Text files: {len(text_files)}")
    print(f"  Expected: {num_queries} each")

    if len(json_files) == num_queries and len(text_files) == num_queries:
        print(f"  ✓ All log files generated correctly")
    else:
        print(f"  ⚠️ Warning: Missing log files")

    export_files = list(export_dir.glob("*"))
    print(f"\nExport files: {len(export_files)}")
    for f in sorted(export_files):
        size = f.stat().st_size / 1024  # KB
        print(f"  - {f.name} ({size:.1f} KB)")

    # Final summary
    print(f"\n{'='*80}")
    print("🎉 STRESS TEST COMPLETE!")
    print(f"{'='*80}")

    print(f"\n✅ Successfully processed {completed} queries")
    print(f"✅ Generated {len(json_files) + len(text_files)} log files")
    print(f"✅ Created {len(export_files)} export files")
    print(f"✅ Throughput: {num_queries / total_time:.2f} queries/second")
    print(f"✅ Success rate: {completed / num_queries * 100:.1f}%")
    print(f"✅ Tracking overhead: {overhead_percentage:.1f}%")

    print(f"\n📁 Results location: {output_dir}/")
    print(f"📁 Exports location: {export_dir}/")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

