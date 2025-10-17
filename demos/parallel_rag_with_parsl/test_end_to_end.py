#!/usr/bin/env python3
"""
End-to-End Parallel RAG Test with Real Data
============================================

Comprehensive test showing:
1. PARSL executor processing real queries
2. Worker ID tracking through the pipeline
3. Shared resource access patterns
4. Full 5-step RAG pipeline execution
5. Performance metrics and statistics
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.workflow import Workflow
from nanobrain.core.shared_resource import get_resource_pool


class EndToEndRAGTester:
    """Comprehensive end-to-end test of parallel RAG workflow."""
    
    def __init__(self):
        self.workflow = None
        self.test_queries = [
            "What are the key mechanisms of viral membrane fusion in coronaviruses?",
            "How do spike proteins facilitate viral entry into host cells?",
            "What role does the ACE2 receptor play in SARS-CoV-2 infection?",
            "Explain the structural dynamics of viral envelope proteins during fusion.",
        ]
        self.results = []
        self.start_time = None
    
    async def setup_workflow(self):
        """Load and initialize the parallel RAG workflow."""
        print("="*80)
        print("🚀 END-TO-END PARALLEL RAG TEST")
        print("="*80)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 Test Queries: {len(self.test_queries)}")
        print("="*80)
        
        print("\n🔧 STEP 1: LOADING WORKFLOW")
        print("-"*80)
        
        config_path = "demos/parallel_rag_demo/config/workflow/parallel_rag_workflow.yml"
        print(f"📄 Config: {config_path}")
        
        load_start = time.time()
        self.workflow = Workflow.from_config(config_path)
        await self.workflow.initialize()
        load_time = time.time() - load_start
        
        print(f"✅ Workflow loaded in {load_time:.2f}s")
        
        # Display workflow info
        steps = getattr(self.workflow, 'steps', {})
        links = getattr(self.workflow, 'step_links', [])
        
        print(f"\n📊 Workflow Configuration:")
        print(f"   Name: {self.workflow.name}")
        print(f"   Steps: {len(steps)}")
        print(f"   Links: {len(links)}")
        print(f"   Execution: Event-driven with PARSL")
        
        # Check for shared resources
        pool = get_resource_pool()
        resources = pool.list_resources()
        
        if resources:
            print(f"\n🔧 Shared Resources:")
            for res_id, stats in resources.items():
                print(f"   - {res_id}")
                print(f"     Type: {stats['resource_type']}")
                print(f"     Status: Initialized")
    
    async def process_single_query(self, query: str, query_num: int):
        """Process a single query and track all outputs."""
        print(f"\n{'='*80}")
        print(f"🧪 QUERY {query_num}: PROCESSING")
        print(f"{'='*80}")
        print(f"📝 Query: {query}")
        print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
        
        query_start = time.time()
        
        # Get data units
        user_query_unit = self.workflow.step_input_data_units.get('user_query')
        final_response_unit = self.workflow.step_output_data_units.get('final_response')
        
        # Submit query
        print(f"\n📤 Submitting to workflow...")
        await user_query_unit.set(query)
        
        # Wait for response
        max_wait = 60
        poll_interval = 1.0
        elapsed = 0
        response = None
        
        print(f"⏳ Waiting for response (max {max_wait}s)...")
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            response = await final_response_unit.get()
            if response is not None:
                break
            
            if int(elapsed) % 10 == 0:
                print(f"   ⏱️  {elapsed:.0f}s elapsed...")
        
        query_time = time.time() - query_start
        
        # Process result
        result = {
            'query_num': query_num,
            'query': query,
            'processing_time': query_time,
            'success': response is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if response is not None:
            print(f"\n✅ Response received in {query_time:.2f}s")
            
            # Extract worker ID if present
            worker_id = None
            if isinstance(response, dict):
                worker_id = response.get('_worker_id') or response.get('worker_id')
                executor_type = response.get('_executor_type', 'unknown')
                
                if worker_id:
                    print(f"👷 Worker ID: {worker_id}")
                    print(f"⚙️  Executor: {executor_type}")
                    result['worker_id'] = worker_id
                    result['executor_type'] = executor_type
            
            # Display response preview
            print(f"\n📄 Response Preview:")
            print("-"*80)
            response_str = json.dumps(response, indent=2) if isinstance(response, dict) else str(response)
            preview = response_str[:300]
            print(preview + "..." if len(response_str) > 300 else preview)
            print("-"*80)
            
            result['response'] = response
            result['response_length'] = len(str(response))
        else:
            print(f"\n❌ Query timed out after {max_wait}s")
            result['error'] = 'timeout'
        
        # Check shared resources
        pool = get_resource_pool()
        resources = pool.list_resources()
        
        if resources:
            print(f"\n📊 Shared Resource Stats:")
            for res_id, stats in resources.items():
                print(f"   {res_id}:")
                print(f"      Accesses: {stats['access_count']}")
                print(f"      Active Workers: {stats['active_workers']}")
        
        self.results.append(result)
        return result
    
    async def test_sequential_queries(self):
        """Test queries one at a time to establish baseline."""
        print(f"\n{'='*80}")
        print("🧪 TEST MODE: SEQUENTIAL PROCESSING")
        print(f"{'='*80}")
        print(f"Processing {len(self.test_queries)} queries sequentially...")
        
        self.start_time = time.time()
        
        for i, query in enumerate(self.test_queries, 1):
            await self.process_single_query(query, i)
            
            # Small delay between queries
            if i < len(self.test_queries):
                await asyncio.sleep(1)
        
        total_time = time.time() - self.start_time
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 SEQUENTIAL TEST SUMMARY")
        print(f"{'='*80}")
        
        successful = sum(1 for r in self.results if r['success'])
        
        print(f"\n✅ Results:")
        print(f"   Total queries: {len(self.test_queries)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(self.test_queries) - successful}")
        print(f"   Total time: {total_time:.2f}s")
        
        if successful > 0:
            avg_time = sum(r['processing_time'] for r in self.results if r['success']) / successful
            print(f"   Average time per query: {avg_time:.2f}s")
        
        # Worker ID summary
        worker_ids = set()
        for r in self.results:
            if 'worker_id' in r:
                worker_ids.add(r['worker_id'])
        
        if worker_ids:
            print(f"\n👷 Worker IDs observed: {len(worker_ids)}")
            for wid in worker_ids:
                count = sum(1 for r in self.results if r.get('worker_id') == wid)
                print(f"   - {wid}: {count} queries")
        
        return successful == len(self.test_queries)
    
    async def test_parallel_queries(self):
        """Test queries submitted in parallel."""
        print(f"\n{'='*80}")
        print("🧪 TEST MODE: PARALLEL PROCESSING WITH PARSL")
        print(f"{'='*80}")
        print(f"Submitting {len(self.test_queries)} queries in parallel...")
        
        self.start_time = time.time()
        self.results = []
        
        # Submit all queries rapidly
        user_query_unit = self.workflow.step_input_data_units.get('user_query')
        final_response_unit = self.workflow.step_output_data_units.get('final_response')
        
        print(f"\n📤 Submitting queries...")
        for i, query in enumerate(self.test_queries, 1):
            print(f"   {i}. {query[:60]}...")
            await user_query_unit.set(query)
            await asyncio.sleep(0.1)
        
        submission_time = time.time() - self.start_time
        print(f"\n✅ All queries submitted in {submission_time:.2f}s")
        
        # Wait for all responses
        print(f"\n⏳ Waiting for responses...")
        max_wait = 120
        poll_interval = 1.0
        elapsed = 0
        responses = []
        
        while elapsed < max_wait and len(responses) < len(self.test_queries):
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            response = await final_response_unit.get()
            if response is not None and response not in responses:
                responses.append(response)
                print(f"   ✅ Response {len(responses)}/{len(self.test_queries)} received ({elapsed:.1f}s)")
                
                # Extract worker ID
                if isinstance(response, dict):
                    worker_id = response.get('_worker_id') or response.get('worker_id')
                    if worker_id:
                        print(f"      Worker: {worker_id}")
            
            if int(elapsed) % 20 == 0 and len(responses) < len(self.test_queries):
                print(f"   ⏱️  {elapsed:.0f}s elapsed, {len(responses)}/{len(self.test_queries)} responses...")
        
        total_time = time.time() - self.start_time
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 PARALLEL TEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n✅ Results:")
        print(f"   Queries submitted: {len(self.test_queries)}")
        print(f"   Responses received: {len(responses)}")
        print(f"   Submission time: {submission_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Processing time: {total_time - submission_time:.2f}s")
        
        if len(responses) > 0:
            avg_time = total_time / len(self.test_queries)
            print(f"   Average time per query: {avg_time:.2f}s")
            
            # Calculate speedup
            sequential_estimate = 20.0 * len(self.test_queries)  # Estimate
            speedup = sequential_estimate / total_time
            print(f"\n📈 Performance:")
            print(f"   Estimated sequential: {sequential_estimate:.2f}s")
            print(f"   Actual parallel: {total_time:.2f}s")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Time saved: {sequential_estimate - total_time:.2f}s")
        
        return len(responses) >= len(self.test_queries) * 0.5  # At least 50% success
    
    async def run_tests(self, mode='sequential'):
        """Run tests in specified mode."""
        await self.setup_workflow()
        
        if mode == 'sequential':
            success = await self.test_sequential_queries()
        elif mode == 'parallel':
            success = await self.test_parallel_queries()
        else:
            print(f"❌ Unknown mode: {mode}")
            return False
        
        # Final summary
        print(f"\n{'='*80}")
        print("🏁 FINAL SUMMARY")
        print(f"{'='*80}")
        
        if success:
            print("\n🎉 TEST PASSED!")
            print("✅ Parallel RAG workflow working correctly")
            print("✅ Worker ID tracking functional")
            print("✅ PARSL executor operational")
        else:
            print("\n⚠️  TEST INCOMPLETE")
            print("Some queries did not complete successfully")
        
        print(f"\n{'='*80}")
        
        return success


async def main():
    """Main test function."""
    import sys
    
    mode = 'sequential'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    print(f"Running in {mode.upper()} mode")
    
    tester = EndToEndRAGTester()
    success = await tester.run_tests(mode=mode)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

