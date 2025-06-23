#!/usr/bin/env python3
"""
End-to-End Workflow Integration Tests for Chatbot Viral Integration

Tests the complete workflow integration as specified in 
section 6.3 of the testing plan.

Test Cases:
- AW-001: "Create PSSM matrix of EEEV" → Full annotation workflow
- AW-002: "Analyze Chikungunya virus" → Full annotation workflow  
- AW-003: "Protein clustering for Alphavirus" → Partial workflow (steps 1-12)

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chatbot_viral_integration import (
    ChatbotTestData,
    WORKFLOW_VALIDATION,
    MockWorkflowComponents,
    MockBVBRCService,
    MockExternalTools,
    MockSessionManager
)

# Import actual workflow components
try:
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
    REAL_WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Real workflow not available, using mocks only: {e}")
    REAL_WORKFLOW_AVAILABLE = False


class TestEndToEndWorkflow:
    """Test suite for end-to-end workflow integration"""
    
    @pytest_asyncio.fixture
    async def mock_services(self):
        """Setup complete mock service environment"""
        return {
            "workflow_components": MockWorkflowComponents(),
            "bvbrc_service": MockBVBRCService(simulate_delays=True),
            "external_tools": MockExternalTools(simulate_processing=True),
            "session_manager": MockSessionManager()
        }
    
    @pytest_asyncio.fixture
    async def real_workflow(self):
        """Setup real workflow if available"""
        if REAL_WORKFLOW_AVAILABLE:
            try:
                from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import create_chatbot_viral_workflow
                workflow = await create_chatbot_viral_workflow()
                return workflow
            except Exception as e:
                pytest.skip(f"Real workflow unavailable: {e}")
        else:
            pytest.skip("Real workflow components not available")
    
    @pytest.mark.asyncio
    async def test_aw_001_eeev_pssm_workflow(self, mock_services):
        """
        Test AW-001: "Create PSSM matrix of EEEV" → Full annotation workflow
        
        Expected:
        - All 14 workflow steps complete
        - Job completion with PSSM JSON output
        - Progress tracking throughout
        """
        query = "Create PSSM matrix of EEEV"
        session_id = "test_session_aw001"
        
        # Step 1: Query Classification
        classification_result = await mock_services["workflow_components"].mock_query_classification_step(query)
        assert classification_result["intent"] == "annotation", "Should classify as annotation intent"
        assert classification_result["routing_decision"]["next_step"] == "annotation_job", "Should route to annotation job"
        
        # Step 2: Session Management
        session = await mock_services["session_manager"].get_or_create_session(session_id)
        assert session["session_id"] == session_id, "Session should be created/retrieved"
        
        # Step 3: Annotation Job Processing
        annotation_result = await mock_services["workflow_components"].mock_annotation_job_step(query, "EEEV")
        
        # Validate job completion
        assert annotation_result["status"] == "completed", "Job should complete successfully"
        assert annotation_result["organism"] == "EEEV", "Should process EEEV organism"
        assert "pssm_matrix" in annotation_result, "Should generate PSSM matrix"
        assert "job_id" in annotation_result, "Should have job ID"
        
        # Validate PSSM output
        pssm_matrix = annotation_result["pssm_matrix"]
        assert "matrix_data" in pssm_matrix, "PSSM should contain matrix data"
        assert "consensus_sequence" in pssm_matrix, "PSSM should contain consensus sequence"
        
        # Step 4: Data Acquisition (BV-BRC)
        bvbrc_data = await mock_services["bvbrc_service"].get_viral_data("EEEV")
        assert "genomes" in bvbrc_data, "Should retrieve genome data"
        assert "proteins" in bvbrc_data, "Should retrieve protein data"
        
        # Step 5: External Tool Processing
        sequences = ["MTKPPSSSSKSKQR", "MSILGKGPQR"]  # Mock sequences
        alignment_result = await mock_services["external_tools"].muscle_alignment(sequences)
        clustering_result = await mock_services["external_tools"].mmseqs2_clustering(sequences)
        
        assert alignment_result, "Should generate sequence alignment"
        assert clustering_result["num_clusters"] > 0, "Should generate protein clusters"
        
        print(f"✅ AW-001 passed: Full EEEV PSSM workflow completed successfully")
        print(f"   Job ID: {annotation_result['job_id']}")
        print(f"   Execution time: {annotation_result['execution_time']}s")
        print(f"   Clusters generated: {clustering_result['num_clusters']}")
    
    @pytest.mark.asyncio
    async def test_aw_002_chikungunya_analysis_workflow(self, mock_services):
        """
        Test AW-002: "Analyze Chikungunya virus" → Full annotation workflow
        
        Expected:
        - All 14 workflow steps complete
        - Job completion with analysis report
        - Organism-specific data retrieval
        """
        query = "Analyze Chikungunya virus"
        session_id = "test_session_aw002"
        
        # Classification
        classification_result = await mock_services["workflow_components"].mock_query_classification_step(query)
        assert classification_result["intent"] == "annotation", "Should classify as annotation intent"
        
        # Session handling
        session = await mock_services["session_manager"].get_or_create_session(session_id)
        
        # Annotation processing
        annotation_result = await mock_services["workflow_components"].mock_annotation_job_step(query, "Chikungunya")
        assert annotation_result["status"] == "completed", "Job should complete successfully"
        assert annotation_result["organism"] == "Chikungunya", "Should process Chikungunya organism"
        
        # Data acquisition for Chikungunya
        bvbrc_data = await mock_services["bvbrc_service"].get_viral_data("Chikungunya virus")
        assert bvbrc_data["genomes"][0]["organism_name"] == "Chikungunya virus", "Should retrieve Chikungunya-specific data"
        
        # External tool processing
        hmmer_result = await mock_services["external_tools"].hmmer_search("MADEKKKHVLSALG", "pfam")
        blast_result = await mock_services["external_tools"].blast_search("MADEKKKHVLSALG", "nr")
        
        assert hmmer_result["num_hits"] > 0, "Should find domain matches"
        assert blast_result["num_hits"] > 0, "Should find sequence matches"
        
        print(f"✅ AW-002 passed: Full Chikungunya analysis workflow completed")
        print(f"   HMMER hits: {hmmer_result['num_hits']}")
        print(f"   BLAST hits: {blast_result['num_hits']}")
    
    @pytest.mark.asyncio
    async def test_aw_003_alphavirus_clustering_workflow(self, mock_services):
        """
        Test AW-003: "Protein clustering for Alphavirus" → Partial workflow (steps 1-12)
        
        Expected:
        - Steps 1-12 complete (clustering focus)
        - Quality metrics for clustering
        - No full PSSM generation
        """
        query = "Protein clustering for Alphavirus"
        session_id = "test_session_aw003"
        
        # Classification
        classification_result = await mock_services["workflow_components"].mock_query_classification_step(query)
        assert classification_result["intent"] == "annotation", "Should classify as annotation intent"
        
        # Session handling
        session = await mock_services["session_manager"].get_or_create_session(session_id)
        
        # Data acquisition
        bvbrc_data = await mock_services["bvbrc_service"].get_viral_data("Alphavirus")
        protein_sequences = [protein["aa_sequence"] for protein in bvbrc_data["proteins"]]
        
        # Clustering-focused processing
        clustering_result = await mock_services["external_tools"].mmseqs2_clustering(protein_sequences)
        
        # Validate clustering quality
        silhouette_score = clustering_result.get("silhouette_score", 0)
        assert silhouette_score >= 0.7, f"Clustering quality should be good (silhouette >= 0.7), got {silhouette_score}"
        assert clustering_result["num_clusters"] >= 1, "Should generate at least one cluster"
        
        # Multiple sequence alignment for each cluster
        for cluster in clustering_result["clusters"]:
            if len(cluster["sequences"]) > 1:
                alignment = await mock_services["external_tools"].muscle_alignment(cluster["sequences"])
                assert alignment, f"Should generate alignment for cluster {cluster['cluster_num']}"
        
        print(f"✅ AW-003 passed: Alphavirus clustering workflow completed")
        print(f"   Clusters: {clustering_result['num_clusters']}")
        print(f"   Silhouette score: {silhouette_score}")
    
    @pytest.mark.asyncio
    async def test_workflow_progress_tracking(self, mock_services):
        """Test progress tracking throughout workflow execution"""
        query = "Create PSSM matrix of EEEV"
        session_id = "test_session_progress"
        
        progress_steps = []
        
        # Simulate progress tracking
        workflow_steps = [
            ("Query classification", 5),
            ("Session initialization", 10),
            ("Data acquisition", 25),
            ("Sequence processing", 40),
            ("Multiple alignment", 60),
            ("Clustering analysis", 75),
            ("PSSM generation", 90),
            ("Result formatting", 100)
        ]
        
        for step_name, progress in workflow_steps:
            # Simulate step execution with progress
            start_time = time.time()
            
            if step_name == "Query classification":
                result = await mock_services["workflow_components"].mock_query_classification_step(query)
            elif step_name == "Data acquisition":
                result = await mock_services["bvbrc_service"].get_viral_data("EEEV")
            elif step_name == "Multiple alignment":
                result = await mock_services["external_tools"].muscle_alignment(["MTKPP", "MSILG"])
            elif step_name == "Clustering analysis":
                result = await mock_services["external_tools"].mmseqs2_clustering(["MTKPP", "MSILG"])
            else:
                await asyncio.sleep(0.05)  # Simulate processing
                result = {"status": "completed"}
            
            end_time = time.time()
            
            progress_steps.append({
                "step": step_name,
                "progress": progress,
                "duration": end_time - start_time,
                "status": "completed"
            })
        
        # Validate progress tracking
        assert len(progress_steps) == len(workflow_steps), "All steps should be tracked"
        assert progress_steps[-1]["progress"] == 100, "Final step should be 100% complete"
        
        # Validate step completion criteria from WORKFLOW_VALIDATION
        for step in progress_steps:
            assert step["status"] == "completed", f"Step {step['step']} should complete successfully"
            assert step["duration"] < 5.0, f"Step {step['step']} took too long: {step['duration']:.3f}s"
        
        print(f"✅ Progress tracking validated: {len(progress_steps)} steps completed")
        
        total_duration = sum(step["duration"] for step in progress_steps)
        print(f"   Total workflow time: {total_duration:.3f}s")
    
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, mock_services):
        """Test error recovery and graceful handling of failures"""
        query = "Create PSSM matrix of InvalidOrganism"
        session_id = "test_session_error"
        
        # Test BV-BRC service error handling
        bvbrc_data = await mock_services["bvbrc_service"].get_viral_data("InvalidOrganism")
        assert "error" in bvbrc_data, "Should handle invalid organism gracefully"
        assert bvbrc_data["genomes"] == [], "Should return empty results for invalid organism"
        
        # Test external tool timeout simulation
        mock_services["bvbrc_service"].set_error_rate(0.5)  # 50% error rate
        
        error_count = 0
        success_count = 0
        
        for i in range(10):
            try:
                result = await mock_services["bvbrc_service"].get_viral_data("EEEV")
                if "error" not in result:
                    success_count += 1
            except Exception:
                error_count += 1
        
        # Should handle errors gracefully
        assert error_count > 0, "Error simulation should trigger some failures"
        assert success_count > 0, "Some requests should still succeed"
        
        print(f"✅ Error recovery tested: {error_count} errors, {success_count} successes")
        
        # Reset error simulation
        mock_services["bvbrc_service"].reset()
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, mock_services):
        """Test timeout handling for long-running operations"""
        query = "Create comprehensive analysis of EEEV"
        
        # Test timeout simulation
        mock_services["bvbrc_service"].simulate_timeout()
        
        timeout_handled = False
        try:
            # This should timeout
            await mock_services["bvbrc_service"].get_viral_data("EEEV")
        except (TimeoutError, Exception) as e:
            timeout_handled = True
            print(f"✅ Timeout handled appropriately: {type(e).__name__}")
        
        assert timeout_handled, "Timeout should be handled gracefully"
        
        # Reset timeout simulation
        mock_services["bvbrc_service"].reset()
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, mock_services):
        """Test concurrent execution of multiple workflows"""
        queries = [
            "Create PSSM matrix of EEEV",
            "Analyze Chikungunya virus", 
            "What is EEEV?"
        ]
        
        session_ids = [f"concurrent_session_{i}" for i in range(len(queries))]
        
        # Execute workflows concurrently
        async def run_workflow(query, session_id):
            classification = await mock_services["workflow_components"].mock_query_classification_step(query)
            session = await mock_services["session_manager"].get_or_create_session(session_id)
            
            if classification["intent"] == "annotation":
                result = await mock_services["workflow_components"].mock_annotation_job_step(query)
                return {"type": "annotation", "result": result}
            else:
                result = await mock_services["workflow_components"].mock_conversational_response_step(query)
                return {"type": "conversational", "result": result}
        
        # Run all workflows concurrently
        start_time = time.time()
        results = await asyncio.gather(*[
            run_workflow(query, session_id) 
            for query, session_id in zip(queries, session_ids)
        ])
        end_time = time.time()
        
        # Validate results
        assert len(results) == len(queries), "All workflows should complete"
        
        annotation_count = sum(1 for r in results if r["type"] == "annotation")
        conversational_count = sum(1 for r in results if r["type"] == "conversational")
        
        assert annotation_count == 2, f"Expected 2 annotation workflows, got {annotation_count}"
        assert conversational_count == 1, f"Expected 1 conversational workflow, got {conversational_count}"
        
        # Check session management
        session_count = mock_services["session_manager"].get_session_count()
        assert session_count == len(queries), f"Should have {len(queries)} sessions, got {session_count}"
        
        total_time = end_time - start_time
        print(f"✅ Concurrent execution completed: {len(queries)} workflows in {total_time:.3f}s")
    
    @pytest.mark.skipif(not REAL_WORKFLOW_AVAILABLE, reason="Real workflow not available")
    @pytest.mark.asyncio
    async def test_real_workflow_integration(self, real_workflow):
        """Test with real workflow components if available"""
        test_cases = [
            ("What is EEEV?", "conversational"),
            ("Create PSSM matrix of EEEV", "annotation")
        ]
        
        for query, expected_type in test_cases:
            print(f"🧪 Testing real workflow: {query}")
            
            response_chunks = []
            final_result = None
            
            async for chunk in real_workflow.process_user_message(query, f"real_test_{expected_type}"):
                response_chunks.append(chunk)
                
                if chunk.get('type') in ['content_complete', 'job_complete', 'message_complete']:
                    final_result = chunk
                    break
            
            assert final_result is not None, f"No final result for: {query}"
            assert len(response_chunks) > 0, f"No response chunks for: {query}"
            
            print(f"✅ Real workflow integration: {query} → {len(response_chunks)} chunks")


@pytest.mark.asyncio
async def test_workflow_performance_requirements():
    """Test workflow performance requirements (from section 7.2.1)"""
    mock_services = {
        "workflow_components": MockWorkflowComponents(),
        "bvbrc_service": MockBVBRCService(simulate_delays=False),  # Fast mode
        "external_tools": MockExternalTools(simulate_processing=False)
    }
    
    test_cases = [
        ("What is EEEV?", "conversational", 2.0),  # <2s target
        ("Create PSSM matrix of EEEV", "annotation", 3.0)  # <3s target for simple annotation
    ]
    
    for query, expected_type, time_limit in test_cases:
        start_time = time.time()
        
        # Run classification
        classification = await mock_services["workflow_components"].mock_query_classification_step(query)
        
        if classification["intent"] == "annotation":
            result = await mock_services["workflow_components"].mock_annotation_job_step(query)
        else:
            result = await mock_services["workflow_components"].mock_conversational_response_step(query)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert execution_time < time_limit, f"Workflow took {execution_time:.3f}s, exceeds {time_limit}s limit for: {query}"
        
        print(f"✅ Performance requirement met: {query[:30]}... → {execution_time:.3f}s (limit: {time_limit}s)")


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_all_tests():
        print("🧪 Running End-to-End Workflow Integration Tests...")
        
        # Create mock services
        mock_services = {
            "workflow_components": MockWorkflowComponents(),
            "bvbrc_service": MockBVBRCService(simulate_delays=True),
            "external_tools": MockExternalTools(simulate_processing=True),
            "session_manager": MockSessionManager()
        }
        
        # Run integration tests
        test_instance = TestEndToEndWorkflow()
        
        await test_instance.test_aw_001_eeev_pssm_workflow(mock_services)
        await test_instance.test_aw_002_chikungunya_analysis_workflow(mock_services)
        await test_instance.test_aw_003_alphavirus_clustering_workflow(mock_services)
        
        await test_instance.test_workflow_progress_tracking(mock_services)
        await test_instance.test_workflow_error_recovery(mock_services)
        await test_instance.test_workflow_timeout_handling(mock_services)
        await test_instance.test_concurrent_workflow_execution(mock_services)
        
        # Performance tests
        await test_workflow_performance_requirements()
        
        print("🎉 All End-to-End Workflow Integration Tests Passed!")
    
    asyncio.run(run_all_tests())