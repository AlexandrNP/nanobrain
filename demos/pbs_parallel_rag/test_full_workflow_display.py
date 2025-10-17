#!/usr/bin/env python3
"""
Full Workflow Display Test - Shows Enhanced Prompts and Final Responses
========================================================================

This test processes queries through the entire RAG workflow and displays:
1. Original query
2. Enhanced prompt (from PARSL step)
3. Final response (from complete workflow)
4. Worker ID tracking
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.workflow import Workflow


async def test_full_workflow_with_display():
    """Test full workflow and display all intermediate and final outputs."""
    
    print("="*80)
    print("🚀 FULL WORKFLOW TEST WITH COMPLETE OUTPUT DISPLAY")
    print("="*80)
    
    # Load workflow
    print("\n🔧 Loading workflow...")
    # Use absolute path relative to this script's location
    script_dir = Path(__file__).parent
    config_path = script_dir / "config" / "workflow" / "parallel_rag_workflow.yml"
    print(f"📄 Config path: {config_path}")
    print(f"📁 Config exists: {config_path.exists()}")
    workflow = Workflow.from_config(str(config_path))
    await workflow.initialize()
    print("✅ Workflow loaded")
    
    # Test queries
    test_queries = [
        "What are the key mechanisms of viral membrane fusion?",
        "How do viral proteins interact with host cell membranes?",
    ]
    
    print(f"\n📝 Testing {len(test_queries)} queries through full workflow")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"🔍 QUERY {i}: FULL WORKFLOW PROCESSING")
        print(f"{'='*80}")
        
        print(f"\n📥 Original Query:")
        print(f"   {query}")
        
        # Get data units
        user_query_unit = workflow.step_input_data_units.get('user_query')
        final_response_unit = workflow.step_output_data_units.get('final_response')
        
        # Submit query
        print(f"\n⏳ Submitting to workflow...")
        start_time = time.time()
        await user_query_unit.set(query)
        
        # Wait for final response
        print(f"⏳ Waiting for complete workflow processing...")
        max_wait = 60
        poll_interval = 1.0
        elapsed = 0
        final_response = None
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            final_response = await final_response_unit.get()
            if final_response is not None:
                break
            
            if int(elapsed) % 10 == 0:
                print(f"   ⏱️  {elapsed:.0f}s elapsed...")
        
        processing_time = time.time() - start_time
        
        if final_response is not None:
            print(f"\n✅ Workflow completed in {processing_time:.2f}s")
            
            # Try to get enhanced prompt from first step
            prompt_step = workflow.child_steps.get('prompt_enhancement_step')
            enhanced_prompt = None
            
            if prompt_step and hasattr(prompt_step, 'step_output_data_units'):
                enhanced_unit = prompt_step.step_output_data_units.get('enhanced_query')
                if enhanced_unit:
                    enhanced_prompt = await enhanced_unit.get()
            
            # Display enhanced prompt
            print(f"\n✨ Enhanced Prompt (from PARSL step):")
            print(f"{'─'*80}")
            if enhanced_prompt:
                if isinstance(enhanced_prompt, dict):
                    enhanced_text = enhanced_prompt.get('enhanced_query', str(enhanced_prompt))
                    worker_id = enhanced_prompt.get('_worker_id') or enhanced_prompt.get('worker_id', 'N/A')
                    print(f"👷 Worker ID: {worker_id}")
                    print(f"\n{enhanced_text}")
                else:
                    print(f"{enhanced_prompt}")
            else:
                print("⚠️  Enhanced prompt not available")
            print(f"{'─'*80}")
            
            # Display final response
            print(f"\n📤 Final Response (from complete workflow):")
            print(f"{'─'*80}")
            if isinstance(final_response, dict):
                response_text = final_response.get('response', final_response.get('final_response', str(final_response)))
                print(f"{response_text}")
            else:
                print(f"{final_response}")
            print(f"{'─'*80}")
            
        else:
            print(f"\n❌ Query timed out after {max_wait}s")
        
        # Small delay between queries
        if i < len(test_queries):
            await asyncio.sleep(2)
    
    print(f"\n{'='*80}")
    print("🏁 FULL WORKFLOW TEST COMPLETE")
    print(f"{'='*80}")


async def main():
    """Main test function."""
    try:
        await test_full_workflow_with_display()
        print("\n✅ Test completed successfully")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

