#!/usr/bin/env python3
"""
Test Query Journey Tracking
============================

Simple test to verify query journey tracking works correctly.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.pbs_parallel_rag.models.query_journey import QueryJourney, Document, StepMetadata
from demos.pbs_parallel_rag.journey_logging.journey_logger import QueryJourneyLogger


def test_data_models():
    """Test data models."""
    print("="*80)
    print("TEST 1: Data Models")
    print("="*80)
    
    # Test Document
    doc = Document(
        doc_id="doc_001",
        content="Viral membrane fusion is a critical step in viral infection.",
        relevance_score=0.92,
        metadata={'source': 'pubmed', 'year': 2023}
    )
    
    print(f"\n✓ Created Document: {doc.doc_id}")
    print(f"  Relevance: {doc.relevance_score}")
    
    # Test serialization
    doc_dict = doc.to_dict()
    doc_restored = Document.from_dict(doc_dict)
    assert doc_restored.doc_id == doc.doc_id
    print(f"✓ Document serialization works")
    
    # Test StepMetadata
    metadata = StepMetadata(
        worker_id="worker_abc123",
        instance_id=12345,
        duration=2.34
    )
    
    print(f"\n✓ Created StepMetadata: {metadata.worker_id}")
    
    # Test QueryJourney
    journey = QueryJourney(
        query_id="q_001",
        original_query="What are the key mechanisms of viral membrane fusion?"
    )
    
    journey.enhanced_query = "What are the primary processes and key mechanisms of viral membrane fusion?"
    journey.enhancement_metadata = metadata
    journey.retrieved_documents = [doc]
    journey.final_response = "Based on the analysis, viral membrane fusion involves..."
    journey.mark_complete()
    
    print(f"\n✓ Created QueryJourney: {journey.query_id}")
    print(f"  Status: {journey.status}")
    print(f"  Complete: {journey.is_complete()}")
    
    # Test serialization
    journey_dict = journey.to_dict()
    journey_restored = QueryJourney.from_dict(journey_dict)
    assert journey_restored.query_id == journey.query_id
    print(f"✓ QueryJourney serialization works")
    
    print(f"\n✅ All data model tests passed!")


def test_journey_logger():
    """Test journey logger."""
    print("\n" + "="*80)
    print("TEST 2: Journey Logger")
    print("="*80)
    
    # Create logger
    output_dir = "demos/pbs_parallel_rag/output/test_logs"
    logger = QueryJourneyLogger(output_dir=output_dir, format="both")
    
    print(f"\n✓ Created logger: {output_dir}")
    
    # Start journey
    journey = logger.start_journey(
        query_id="q_test_001",
        original_query="What are the key mechanisms of viral membrane fusion?"
    )
    
    print(f"✓ Started journey: {journey.query_id}")
    
    # Simulate enhancement step
    time.sleep(0.1)
    logger.update_enhancement(
        query_id="q_test_001",
        enhanced_query="What are the primary processes, key mechanisms, and fundamental principles of viral membrane fusion?",
        worker_id="worker_abc123",
        instance_id=4684434320,
        duration=2.34
    )
    
    print(f"✓ Updated enhancement")
    
    # Simulate retrieval step
    time.sleep(0.1)
    documents = [
        Document(
            doc_id="doc_001",
            content="Viral membrane fusion is a critical step in the viral life cycle. The process involves conformational changes in viral fusion proteins that bring the viral and cellular membranes into close proximity.",
            relevance_score=0.92,
            metadata={'source': 'pubmed', 'year': 2023, 'pmid': '12345678'}
        ),
        Document(
            doc_id="doc_002",
            content="The spike protein of SARS-CoV-2 mediates viral entry through receptor binding and membrane fusion. The S2 subunit contains the fusion peptide and heptad repeat regions.",
            relevance_score=0.89,
            metadata={'source': 'pubmed', 'year': 2022, 'pmid': '87654321'}
        ),
        Document(
            doc_id="doc_003",
            content="Membrane fusion proceeds through hemifusion intermediates where the outer leaflets merge before complete fusion pore formation.",
            relevance_score=0.85,
            metadata={'source': 'pubmed', 'year': 2021, 'pmid': '11223344'}
        )
    ]
    
    logger.update_retrieval(
        query_id="q_test_001",
        documents=documents,
        worker_id="worker_abc123",
        instance_id=4684434321,
        duration=1.23
    )
    
    print(f"✓ Updated retrieval ({len(documents)} documents)")
    
    # Simulate generation step
    time.sleep(0.1)
    logger.update_generation(
        query_id="q_test_001",
        final_response="""Based on the analysis of viral membrane fusion mechanisms, the key processes include:

1. **Receptor Binding**: The viral spike protein binds to host cell receptors (e.g., ACE2 for SARS-CoV-2), initiating the fusion process.

2. **Conformational Changes**: Upon receptor binding, the fusion protein undergoes dramatic conformational changes that expose the fusion peptide.

3. **Membrane Hemifusion**: The fusion peptide inserts into the target membrane, bringing viral and cellular membranes into close proximity. The outer leaflets merge first, creating a hemifusion intermediate.

4. **Pore Formation**: The hemifusion intermediate transitions to complete fusion through formation and expansion of a fusion pore, allowing viral genetic material to enter the host cell.

These mechanisms are highly conserved across enveloped viruses and represent critical targets for antiviral therapeutics.""",
        worker_id="worker_abc123",
        instance_id=4684434322,
        duration=3.45
    )
    
    print(f"✓ Updated generation")
    
    # Complete journey
    logger.complete_journey("q_test_001")
    print(f"✓ Completed journey")
    
    # Save journey
    logger.save_journey("q_test_001")
    print(f"✓ Saved journey to files")
    
    # Verify files exist
    json_file = Path(output_dir) / "queries" / "q_test_001.json"
    text_file = Path(output_dir) / "queries" / "q_test_001.txt"
    
    assert json_file.exists(), f"JSON file not found: {json_file}"
    assert text_file.exists(), f"Text file not found: {text_file}"
    
    print(f"✓ Verified files exist:")
    print(f"  - {json_file}")
    print(f"  - {text_file}")
    
    # Get summary
    summary = logger.get_summary()
    print(f"\n✓ Summary:")
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Complete: {summary['complete']}")
    print(f"  Avg time: {summary['avg_total_time']:.2f}s")
    print(f"  Avg relevance: {summary['avg_relevance_score']:.2f}")
    
    print(f"\n✅ All logger tests passed!")
    
    # Print file contents for verification
    print(f"\n" + "="*80)
    print("SAMPLE OUTPUT - JSON FORMAT")
    print("="*80)
    with open(json_file, 'r') as f:
        print(f.read())
    
    print(f"\n" + "="*80)
    print("SAMPLE OUTPUT - TEXT FORMAT")
    print("="*80)
    with open(text_file, 'r') as f:
        print(f.read())


def main():
    """Run all tests."""
    try:
        test_data_models()
        test_journey_logger()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n✅ Data models working correctly")
        print("✅ Journey logger working correctly")
        print("✅ JSON and text output generated")
        print("\nNext step: Integrate with workflow steps")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

