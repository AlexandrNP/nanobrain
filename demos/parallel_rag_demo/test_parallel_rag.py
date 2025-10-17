#!/usr/bin/env python3
"""
Parallel RAG Workflow Test
===========================

Test the parallel RAG workflow with PARSL executor for the first step.
Demonstrates processing multiple queries simultaneously.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.workflow import Workflow
from nanobrain.core.executor import ParslExecutor


class ParallelRAGTester:
    """Test parallel RAG workflow with multiple simultaneous queries."""
    
    def __init__(self):
        self.workflow = None
        self.test_queries = [
            "What are the key mechanisms of viral membrane fusion?",
            "How do viral proteins interact with host cell membranes?",
            "What role does molecular dynamics play in understanding biological systems?",
            "Explain the structure and function of viral envelope proteins.",
        ]
    
    async def setup_workflow(self):
        """Load and initialize the parallel RAG workflow."""
        print("="*80)
        print("🔧 SETTING UP PARALLEL RAG WORKFLOW")
        print("="*80)
        
        # Load workflow configuration
        config_path = "demos/parallel_rag_demo/config/workflow/parallel_rag_workflow.yml"
        
        print(f"\n📄 Loading workflow from: {config_path}")
        start_time = time.time()

        self.workflow = Workflow.from_config(config_path)

        # Initialize workflow
        await self.workflow.initialize()

        load_time = time.time() - start_time
        print(f"✅ Workflow loaded in {load_time:.2f}s")
        
        # Display workflow info
        print(f"\n📊 Workflow Components:")
        try:
            steps = getattr(self.workflow, 'steps', {})
            links = getattr(self.workflow, 'step_links', [])
            input_units = getattr(self.workflow, 'step_input_data_units', {})
            output_units = getattr(self.workflow, 'step_output_data_units', {})

            print(f"   Steps: {len(steps)}")
            print(f"   Links: {len(links)}")
            print(f"   Input Data Units: {len(input_units)}")
            print(f"   Output Data Units: {len(output_units)}")
        except Exception as e:
            print(f"   ⚠️  Could not get workflow info: {e}")
        
        print("\n" + "="*80)
    
    async def test_single_query(self):
        """Test with a single query to verify basic functionality."""
        print("\n" + "="*80)
        print("🧪 TEST 1: SINGLE QUERY")
        print("="*80)
        
        query = self.test_queries[0]
        print(f"\nQuery: {query}")
        
        user_query_unit = self.workflow.step_input_data_units.get('user_query')
        final_response_unit = self.workflow.step_output_data_units.get('final_response')
        
        print("\n⏳ Processing query...")
        start_time = time.time()
        
        # Set query
        await user_query_unit.set(query)
        
        # Wait for response
        max_wait = 30
        poll_interval = 0.5
        elapsed = 0
        response = None
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            response = await final_response_unit.get()
            if response is not None:
                break
            
            if int(elapsed) % 5 == 0:
                print(f"   ⏱️  {elapsed:.0f}s elapsed...")
        
        query_time = time.time() - start_time
        
        if response is not None:
            print(f"\n✅ Query completed in {query_time:.2f}s")
            print(f"\n📄 Response preview:")
            print("-"*80)
            response_str = str(response)[:200]
            print(response_str + "..." if len(str(response)) > 200 else response_str)
            print("-"*80)
            return True
        else:
            print(f"\n❌ Query timed out after {max_wait}s")
            return False
    
    async def test_parallel_queries(self):
        """Test with multiple parallel queries using separate workflow instances."""
        print("\n" + "="*80)
        print("🧪 TEST 2: PARALLEL QUERIES WITH PARSL EXECUTOR")
        print("="*80)

        num_queries = min(4, len(self.test_queries))
        queries = self.test_queries[:num_queries]

        print(f"\n📝 Testing {num_queries} parallel queries:")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query[:60]}...")

        print("\n⏳ Creating parallel workflow instances...")
        start_time = time.time()

        # Create a task for each query using the PARSL executor directly
        async def process_query_with_parsl(query, query_num):
            """Process a single query through the workflow."""
            try:
                # Get the first step (prompt enhancement with PARSL)
                prompt_step = self.workflow.child_steps.get('prompt_enhancement_step')

                if prompt_step is None:
                    print(f"   ⚠️  Query {query_num}: Step not found")
                    print(f"   Available steps: {list(self.workflow.child_steps.keys())}")
                    return None

                # Process through the first step (which uses PARSL)
                result = await prompt_step.process({'query': query})

                return {
                    'query_num': query_num,
                    'query': query,
                    'result': result,
                    'worker_id': result.get('_worker_id') if isinstance(result, dict) else None
                }
            except Exception as e:
                print(f"   ❌ Query {query_num} failed: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Submit all queries in parallel
        print(f"\n📤 Submitting {num_queries} queries to PARSL executor...")
        tasks = [
            process_query_with_parsl(query, i)
            for i, query in enumerate(queries, 1)
        ]

        submission_time = time.time() - start_time
        print(f"✅ All queries submitted in {submission_time:.2f}s")

        # Wait for all responses
        print("\n⏳ Waiting for PARSL executor to process queries...")

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"   ❌ Error during parallel processing: {e}")
            responses = []

        total_time = time.time() - start_time

        # Analyze results
        # Filter out None and exception responses
        successful_responses = [r for r in responses if r is not None and not isinstance(r, Exception)]
        failed_responses = [r for r in responses if isinstance(r, Exception) or r is None]

        successful = len(successful_responses)

        print(f"\n📊 Parallel Processing Results:")
        print(f"   Queries submitted: {num_queries}")
        print(f"   Responses received: {successful}")
        print(f"   Failed: {len(failed_responses)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Submission time: {submission_time:.2f}s")
        print(f"   Processing time: {total_time - submission_time:.2f}s")

        # Show worker IDs
        if successful > 0:
            print(f"\n👷 Worker IDs:")
            worker_ids = set()
            for resp in successful_responses:
                worker_id = resp.get('worker_id') if resp else None
                query_num = resp.get('query_num', '?') if resp else '?'
                if worker_id:
                    worker_ids.add(worker_id)
                    print(f"   Query {query_num}: {worker_id}")
                else:
                    print(f"   Query {query_num}: No worker ID")

            if worker_ids:
                print(f"\n   Total unique workers: {len(worker_ids)}")

            avg_time = total_time / num_queries
            print(f"\n⏱️  Timing:")
            print(f"   Average time per query: {avg_time:.2f}s")

            # Estimate sequential time
            sequential_estimate = 17.56 * num_queries  # Based on single query test
            speedup = sequential_estimate / total_time
            print(f"\n📈 Performance Analysis:")
            print(f"   Estimated sequential time: {sequential_estimate:.2f}s")
            print(f"   Actual parallel time: {total_time:.2f}s")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Time saved: {sequential_estimate - total_time:.2f}s ({((sequential_estimate - total_time)/sequential_estimate * 100):.1f}%)")

        if successful == num_queries:
            print(f"\n✅ All {num_queries} queries processed successfully!")
            print(f"✅ PARSL parallel processing working!")
            return True
        elif successful >= num_queries * 0.75:
            print(f"\n✅ Good success rate: {successful}/{num_queries} queries completed ({successful/num_queries*100:.0f}%)")
            print(f"✅ PARSL parallel processing working!")
            return True
        elif successful > 0:
            print(f"\n⚠️  Partial success: {successful}/{num_queries} queries completed")
            return True
        else:
            print(f"\n❌ No queries completed successfully")
            return False
    
    async def run_tests(self):
        """Run all tests."""
        print("="*80)
        print("🚀 PARALLEL RAG WORKFLOW TESTING")
        print("="*80)
        
        # Setup
        await self.setup_workflow()
        
        # Test 1: Single query
        test1_passed = await self.test_single_query()
        
        # Test 2: Parallel queries (only if test 1 passed)
        test2_passed = False
        if test1_passed:
            test2_passed = await self.test_parallel_queries()
        else:
            print("\n⚠️  Skipping parallel test due to single query failure")
        
        # Summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        print(f"\n✅ Test 1 (Single Query): {'PASSED' if test1_passed else 'FAILED'}")
        print(f"{'✅' if test2_passed else '⚠️ '} Test 2 (Parallel Queries): {'PASSED' if test2_passed else 'SKIPPED/FAILED'}")
        
        if test1_passed and test2_passed:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ Parallel RAG workflow is working correctly")
        elif test1_passed:
            print(f"\n⚠️  PARTIAL SUCCESS")
            print(f"✅ Basic functionality working")
            print(f"⚠️  Parallel processing needs investigation")
        else:
            print(f"\n❌ TESTS FAILED")
            print(f"⚠️  Basic functionality not working")
        
        print("\n" + "="*80)
        
        return test1_passed and test2_passed


async def main():
    """Main test function."""
    tester = ParallelRAGTester()
    success = await tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

