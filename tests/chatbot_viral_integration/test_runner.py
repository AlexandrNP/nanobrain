#!/usr/bin/env python3
"""
Test Runner for Chatbot Viral Integration Testing

Orchestrates execution of all integration tests and generates
comprehensive reports as specified in the testing plan.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chatbot_viral_integration import (
    ChatbotTestData,
    CLASSIFICATION_METRICS,
    CONTENT_QUALITY_CHECKS,
    WORKFLOW_VALIDATION,
    MockWorkflowComponents,
    MockBVBRCService,
    MockExternalTools,
    MockSessionManager
)


@dataclass
class ExecutionResultData:
    """Test result data structure"""
    test_name: str
    test_category: str
    status: str  # "PASSED", "FAILED", "SKIPPED"
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class IntegrationTestRunner:
    """Test runner for chatbot viral integration tests"""
    
    def __init__(self):
        self.results: List[ExecutionResultData] = []
        self.mock_services = self._setup_mock_services()
        
    def _setup_mock_services(self) -> Dict[str, Any]:
        """Setup mock services for testing"""
        return {
            "workflow_components": MockWorkflowComponents(),
            "bvbrc_service": MockBVBRCService(simulate_delays=True),
            "external_tools": MockExternalTools(simulate_processing=True),
            "session_manager": MockSessionManager()
        }
    
    async def run_test(self, test_name: str, test_category: str, test_func, *args, **kwargs) -> ExecutionResultData:
        """Run a single test and capture results"""
        start_time = time.time()
        
        try:
            print(f"🧪 Running {test_name}...")
            await test_func(*args, **kwargs)
            end_time = time.time()
            
            result = ExecutionResultData(
                test_name=test_name,
                test_category=test_category,
                status="PASSED",
                duration=end_time - start_time
            )
            print(f"✅ {test_name} PASSED ({result.duration:.3f}s)")
            
        except Exception as e:
            end_time = time.time()
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            result = ExecutionResultData(
                test_name=test_name,
                test_category=test_category,
                status="FAILED",
                duration=end_time - start_time,
                error_message=error_msg,
                details={"traceback": traceback.format_exc()}
            )
            print(f"❌ {test_name} FAILED ({result.duration:.3f}s): {error_msg}")
        
        self.results.append(result)
        return result
    
    async def run_query_classification_tests(self):
        """Run all query classification tests (Phase 2)"""
        print("\n📋 Phase 2: Query Classification Tests")
        print("=" * 50)
        
        mock_components = self.mock_services["workflow_components"]
        
        # Test QC-001: Conversational EEEV
        await self.run_test(
            "QC-001: Conversational EEEV Query",
            "Query Classification",
            self._test_qc_001,
            mock_components
        )
        
        # Test QC-002: Annotation PSSM
        await self.run_test(
            "QC-002: Annotation PSSM Query",
            "Query Classification", 
            self._test_qc_002,
            mock_components
        )
        
        # Test QC-003: Conversational proteins
        await self.run_test(
            "QC-003: Conversational proteins",
            "Query Classification",
            self._test_qc_003,
            mock_components
        )
        
        # Test QC-004: Annotation analyze
        await self.run_test(
            "QC-004: Annotation analyze",
            "Query Classification",
            self._test_qc_004,
            mock_components
        )
        
        # Test QC-005: Mixed intent
        await self.run_test(
            "QC-005: Mixed intent",
            "Query Classification",
            self._test_qc_005,
            mock_components
        )
        
        # Classification accuracy metrics
        await self.run_test(
            "Classification Accuracy Metrics",
            "Query Classification",
            self._test_classification_accuracy,
            mock_components
        )
    
    async def run_conversational_response_tests(self):
        """Run all conversational response tests (Phase 2)"""
        print("\n💬 Phase 2: Conversational Response Tests")
        print("=" * 50)
        
        mock_components = self.mock_services["workflow_components"]
        
        # Test CR-001: EEEV information
        await self.run_test(
            "CR-001: EEEV Information",
            "Conversational Response",
            self._test_cr_001,
            mock_components
        )
        
        # Test CR-002: Virus transmission
        await self.run_test(
            "CR-002: Virus transmission",
            "Conversational Response",
            self._test_cr_002,
            mock_components
        )
        
        # Test CR-003: PSSM explanation
        await self.run_test(
            "CR-003: PSSM explanation",
            "Conversational Response",
            self._test_cr_003,
            mock_components
        )
        
        # Content quality validation
        await self.run_test(
            "Content Quality Validation",
            "Conversational Response",
            self._test_content_quality,
            mock_components
        )
    
    async def run_workflow_integration_tests(self):
        """Run all workflow integration tests (Phase 2)"""
        print("\n🔄 Phase 2: Workflow Integration Tests")
        print("=" * 50)
        
        # Test AW-001: EEEV PSSM workflow
        await self.run_test(
            "AW-001: EEEV PSSM Workflow",
            "Workflow Integration",
            self._test_aw_001,
            self.mock_services
        )
        
        # Test AW-002: Chikungunya analysis
        await self.run_test(
            "AW-002: Chikungunya Analysis",
            "Workflow Integration",
            self._test_aw_002,
            self.mock_services
        )
        
        # Test AW-003: Alphavirus clustering
        await self.run_test(
            "AW-003: Alphavirus Clustering",
            "Workflow Integration",
            self._test_aw_003,
            self.mock_services
        )
        
        # Progress tracking
        await self.run_test(
            "Progress Tracking",
            "Workflow Integration",
            self._test_progress_tracking,
            self.mock_services
        )
        
        # Error recovery
        await self.run_test(
            "Error Recovery",
            "Workflow Integration", 
            self._test_error_recovery,
            self.mock_services
        )
        
        # Concurrent execution
        await self.run_test(
            "Concurrent Execution",
            "Workflow Integration",
            self._test_concurrent_execution,
            self.mock_services
        )
    
    async def run_performance_tests(self):
        """Run performance tests (Phase 3)"""
        print("\n⚡ Phase 3: Performance Tests")
        print("=" * 50)
        
        # Classification performance
        await self.run_test(
            "Classification Performance",
            "Performance",
            self._test_classification_performance,
            self.mock_services
        )
        
        # Conversational response performance
        await self.run_test(
            "Conversational Response Performance", 
            "Performance",
            self._test_conversational_performance,
            self.mock_services
        )
        
        # Workflow performance
        await self.run_test(
            "Workflow Performance",
            "Performance",
            self._test_workflow_performance,
            self.mock_services
        )
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🎯 Chatbot Viral Integration Test Suite")
        print("=" * 60)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        start_time = time.time()
        
        # Phase 1: Foundation (already done - mock services setup)
        print("✅ Phase 1: Foundation - Mock services ready")
        
        # Phase 2: Core Functionality
        await self.run_query_classification_tests()
        await self.run_conversational_response_tests()
        await self.run_workflow_integration_tests()
        
        # Phase 3: Performance Testing
        await self.run_performance_tests()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate summary report
        self.generate_summary_report(total_duration)
    
    def generate_summary_report(self, total_duration: float):
        """Generate comprehensive test summary report"""
        print("\n📊 Test Summary Report")
        print("=" * 60)
        
        # Count results by category and status
        by_category = {}
        by_status = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0}
        
        for result in self.results:
            # By category
            if result.test_category not in by_category:
                by_category[result.test_category] = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0}
            by_category[result.test_category][result.status] += 1
            
            # By status
            by_status[result.status] += 1
        
        # Overall statistics
        total_tests = len(self.results)
        success_rate = (by_status["PASSED"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        print("📈 Results by Status:")
        for status, count in by_status.items():
            print(f"  {status}: {count}")
        print()
        
        print("📂 Results by Category:")
        for category, counts in by_category.items():
            total_cat = sum(counts.values())
            passed_cat = counts["PASSED"]
            cat_success = (passed_cat / total_cat * 100) if total_cat > 0 else 0
            print(f"  {category}: {passed_cat}/{total_cat} ({cat_success:.1f}%)")
        
        # Failed tests details
        failed_tests = [r for r in self.results if r.status == "FAILED"]
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.error_message}")
        
        # Performance summary
        performance_tests = [r for r in self.results if r.test_category == "Performance"]
        if performance_tests:
            print("\n⚡ Performance Summary:")
            for test in performance_tests:
                status_emoji = "✅" if test.status == "PASSED" else "❌"
                print(f"  {status_emoji} {test.test_name}: {test.duration:.3f}s")
        
        print()
        if by_status["FAILED"] == 0:
            print("🎉 All tests passed successfully!")
        else:
            print(f"⚠️ {by_status['FAILED']} test(s) failed. See details above.")
        
        # Save detailed report to file
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """Save detailed test report to JSON file"""
        report_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASSED"),
            "failed": sum(1 for r in self.results if r.status == "FAILED"),
            "skipped": sum(1 for r in self.results if r.status == "SKIPPED"),
            "total_duration": sum(r.duration for r in self.results),
            "results": [asdict(result) for result in self.results]
        }
        
        report_file = Path(__file__).parent / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    # Individual test implementations
    async def _test_qc_001(self, mock_components):
        """QC-001: What is EEEV? → conversational"""
        query = "What is EEEV?"
        result = await mock_components.mock_query_classification_step(query)
        assert result["intent"] == "conversational"
        assert result["confidence"] >= 0.8
    
    async def _test_qc_002(self, mock_components):
        """QC-002: Create PSSM matrix of EEEV → annotation"""
        query = "Create PSSM matrix of EEEV"
        result = await mock_components.mock_query_classification_step(query)
        assert result["intent"] == "annotation"
        assert result["confidence"] >= 0.8
    
    async def _test_qc_003(self, mock_components):
        """QC-003: Tell me about EEEV proteins → conversational"""
        query = "Tell me about EEEV proteins"
        result = await mock_components.mock_query_classification_step(query)
        assert result["intent"] == "conversational"
        assert result["confidence"] >= 0.6
    
    async def _test_qc_004(self, mock_components):
        """QC-004: Analyze EEEV protein structure → annotation"""
        query = "Analyze EEEV protein structure"
        result = await mock_components.mock_query_classification_step(query)
        assert result["intent"] == "annotation"
        assert result["confidence"] >= 0.7
    
    async def _test_qc_005(self, mock_components):
        """QC-005: What is EEEV and create PSSM? → annotation"""
        query = "What is EEEV and create PSSM?"
        result = await mock_components.mock_query_classification_step(query)
        assert result["intent"] == "annotation"
        assert result["confidence"] >= 0.6
    
    async def _test_classification_accuracy(self, mock_components):
        """Test overall classification accuracy"""
        test_cases = ChatbotTestData.get_classification_test_cases()
        correct = 0
        
        for case in test_cases:
            result = await mock_components.mock_query_classification_step(case["input_query"])
            if result["intent"] == case["expected_intent"]:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= CLASSIFICATION_METRICS["accuracy_threshold"]
    
    async def _test_cr_001(self, mock_components):
        """CR-001: What is EEEV? → Educational content"""
        query = "What is EEEV?"
        result = await mock_components.mock_conversational_response_step(query)
        response = result["response"].lower()
        
        eeev_terms = ["eeev", "eastern equine", "virus"]
        assert any(term in response for term in eeev_terms)
        assert result["requires_markdown"]
    
    async def _test_cr_002(self, mock_components):
        """CR-002: How do viruses spread? → Transmission mechanisms"""
        query = "How do viruses spread?"
        result = await mock_components.mock_conversational_response_step(query)
        response = result["response"].lower()
        
        transmission_terms = ["transmission", "spread", "vector", "mosquito"]
        assert any(term in response for term in transmission_terms)
    
    async def _test_cr_003(self, mock_components):
        """CR-003: What is a PSSM matrix? → Technical explanation"""
        query = "What is a PSSM matrix?"
        result = await mock_components.mock_conversational_response_step(query)
        response = result["response"].lower()
        
        pssm_terms = ["pssm", "matrix", "bioinformatics"]
        assert sum(1 for term in pssm_terms if term in response) >= 2
    
    async def _test_content_quality(self, mock_components):
        """Test content quality validation"""
        query = "What is EEEV?"
        result = await mock_components.mock_conversational_response_step(query)
        response = result["response"]
        
        assert len(response) >= CONTENT_QUALITY_CHECKS["min_length"]
        assert len(response) <= CONTENT_QUALITY_CHECKS["max_length"]
        assert "**" in response or "#" in response  # Markdown formatting
    
    async def _test_aw_001(self, mock_services):
        """AW-001: Create PSSM matrix of EEEV → Full workflow"""
        query = "Create PSSM matrix of EEEV"
        
        # Classification
        classification = await mock_services["workflow_components"].mock_query_classification_step(query)
        assert classification["intent"] == "annotation"
        
        # Annotation job
        annotation = await mock_services["workflow_components"].mock_annotation_job_step(query, "EEEV")
        assert annotation["status"] == "completed"
        assert "pssm_matrix" in annotation
        
        # Data acquisition
        data = await mock_services["bvbrc_service"].get_viral_data("EEEV")
        assert "genomes" in data
    
    async def _test_aw_002(self, mock_services):
        """AW-002: Analyze Chikungunya virus → Full workflow"""
        query = "Analyze Chikungunya virus"
        
        classification = await mock_services["workflow_components"].mock_query_classification_step(query)
        assert classification["intent"] == "annotation"
        
        annotation = await mock_services["workflow_components"].mock_annotation_job_step(query, "Chikungunya")
        assert annotation["status"] == "completed"
        
        data = await mock_services["bvbrc_service"].get_viral_data("Chikungunya virus")
        assert data["genomes"][0]["organism_name"] == "Chikungunya virus"
    
    async def _test_aw_003(self, mock_services):
        """AW-003: Protein clustering for Alphavirus → Partial workflow"""
        query = "Protein clustering for Alphavirus"
        
        classification = await mock_services["workflow_components"].mock_query_classification_step(query)
        assert classification["intent"] == "annotation"
        
        data = await mock_services["bvbrc_service"].get_viral_data("Alphavirus")
        clustering = await mock_services["external_tools"].mmseqs2_clustering(["MSILG"])
        
        assert clustering["silhouette_score"] >= 0.7
        assert clustering["num_clusters"] >= 1
    
    async def _test_progress_tracking(self, mock_services):
        """Test workflow progress tracking"""
        query = "Create PSSM matrix of EEEV"
        
        steps = [
            mock_services["workflow_components"].mock_query_classification_step(query),
            mock_services["bvbrc_service"].get_viral_data("EEEV"),
            mock_services["external_tools"].muscle_alignment(["MTKPP"]),
            mock_services["external_tools"].mmseqs2_clustering(["MTKPP"])
        ]
        
        for step in steps:
            result = await step
            assert result is not None
    
    async def _test_error_recovery(self, mock_services):
        """Test error recovery mechanisms"""
        # Test invalid organism
        data = await mock_services["bvbrc_service"].get_viral_data("InvalidOrganism")
        assert "error" in data
        assert data["genomes"] == []
        
        # Test error rate simulation
        mock_services["bvbrc_service"].set_error_rate(0.5)
        error_count = 0
        
        for _ in range(5):
            try:
                await mock_services["bvbrc_service"].get_viral_data("EEEV")
            except:
                error_count += 1
        
        mock_services["bvbrc_service"].reset()
        # Should have some errors due to 50% error rate
    
    async def _test_concurrent_execution(self, mock_services):
        """Test concurrent workflow execution"""
        queries = [
            "Create PSSM matrix of EEEV",
            "What is EEEV?",
            "Analyze Chikungunya virus"
        ]
        
        async def process_query(query):
            classification = await mock_services["workflow_components"].mock_query_classification_step(query)
            return classification
        
        results = await asyncio.gather(*[process_query(q) for q in queries])
        assert len(results) == len(queries)
        assert all("intent" in r for r in results)
    
    async def _test_classification_performance(self, mock_services):
        """Test classification performance"""
        query = "Create PSSM matrix of EEEV"
        
        start_time = time.time()
        result = await mock_services["workflow_components"].mock_query_classification_step(query)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 1.0  # Should be under 1 second
        assert result["intent"] == "annotation"
    
    async def _test_conversational_performance(self, mock_services):
        """Test conversational response performance"""
        query = "What is EEEV?"
        
        start_time = time.time()
        result = await mock_services["workflow_components"].mock_conversational_response_step(query)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 5.0  # Should be under 5 seconds
        assert "response" in result
    
    async def _test_workflow_performance(self, mock_services):
        """Test overall workflow performance"""
        query = "Create PSSM matrix of EEEV"
        
        start_time = time.time()
        
        classification = await mock_services["workflow_components"].mock_query_classification_step(query)
        annotation = await mock_services["workflow_components"].mock_annotation_job_step(query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 3.0  # Should complete under 3 seconds for mock
        assert classification["intent"] == "annotation"
        assert annotation["status"] == "completed"


async def main():
    """Main test runner entry point"""
    runner = IntegrationTestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())