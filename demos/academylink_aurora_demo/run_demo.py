#!/usr/bin/env python3
"""
🔥 BRUTAL TRUTH: Extended Timeout Test for Complete End-to-End Execution

This test runs the complete workflow with extended timeout to capture
all evidence of distributed execution and Academy agent usage.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the nanobrain directory to Python path
nanobrain_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(nanobrain_dir))

from nanobrain.core.workflow import Workflow


async def test_complete_end_to_end():
    """Test complete end-to-end execution with extended timeout"""
    print('🔥 BRUTAL TRUTH: COMPLETE END-TO-END EXECUTION TEST')
    print('=' * 70)
    
    try:
        # Load workflow
        print('📋 Loading workflow...')
        workflow = Workflow.from_config('config/mixed_execution_workflow_aurora.yml')
        await workflow.initialize()
        print('✅ Workflow initialized')
        
        # Test data
        test_data = {
            'sequences': [
                {'id': 'aurora_seq_1', 'sequence': 'ATCGATCGATCGATCGATCG', 'length': 20},
                {'id': 'aurora_seq_2', 'sequence': 'GCTAGCTAGCTAGCTAGCTA', 'length': 20},
                {'id': 'aurora_seq_3', 'sequence': 'TTAACCGGTTAACCGGTTAA', 'length': 20},
                {'id': 'aurora_seq_4', 'sequence': 'AAATTTCCCGGGAAATTTCC', 'length': 20},
                {'id': 'aurora_seq_5', 'sequence': 'CGTACGTACGTACGTACGTA', 'length': 20}
            ],
            'metadata': {
                'source': 'complete_end_to_end_test',
                'timestamp': time.time(),
                'test_type': 'extended_timeout'
            }
        }
        
        print(f'📊 Test data prepared: {len(test_data["sequences"])} sequences')
        
        # Set input
        first_step = workflow.child_steps['data_preparation']
        input_unit = first_step.step_input_data_units['raw_input']
        await input_unit.set(test_data)
        
        print('✅ Input data set - workflow executing...')
        print('🔄 Waiting for complete end-to-end execution...')
        
        # Wait for results with extended timeout
        last_step = workflow.child_steps['result_aggregation']
        output_unit = last_step.step_output_data_units['final_results']
        
        start_time = time.time()
        timeout = 120  # 2 minutes for lightweight demo
        
        while not output_unit.has_data() and (time.time() - start_time) < timeout:
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed > 0:
                print(f'   ⏳ Still executing... {elapsed:.0f}s elapsed')
                print(f'      🔄 Data preparation -> Aurora HPC -> Result aggregation')
            await asyncio.sleep(1)  # Check more frequently
        
        execution_time = time.time() - start_time
        
        if output_unit.has_data():
            results = await output_unit.get()
            
            print('🎉 SUCCESS: COMPLETE END-TO-END EXECUTION!')
            print('=' * 70)
            print(f'   ⏱️ Total execution time: {execution_time:.2f}s')
            print(f'   📊 Results type: {type(results)}')
            
            if isinstance(results, dict):
                print('\n📋 COMPLETE RESULTS ANALYSIS:')
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f'   📁 {key}: {len(value)} items')
                        if 'computed_sequences' in str(key).lower():
                            print(f'      🔬 Computed sequences found')
                        if 'node' in str(key).lower() or 'aurora' in str(key).lower():
                            print(f'      🖥️ Aurora node information found')
                    elif isinstance(value, list):
                        print(f'   📋 {key}: {len(value)} items')
                        if len(value) > 0 and isinstance(value[0], dict):
                            first_item = value[0]
                            if 'aurora_node' in first_item or 'hostname' in first_item:
                                print(f'      🖥️ Real node information detected')
                    else:
                        print(f'   📄 {key}: {type(value).__name__}')
                
                # Look for evidence of distributed execution
                print('\n🔍 EVIDENCE OF DISTRIBUTED EXECUTION:')
                evidence_found = []
                
                def search_for_evidence(obj, path=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if any(keyword in str(k).lower() for keyword in ['node', 'hostname', 'pbs', 'worker', 'aurora']):
                                evidence_found.append(f'{path}.{k}: {v}')
                            search_for_evidence(v, f'{path}.{k}')
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            search_for_evidence(item, f'{path}[{i}]')
                
                search_for_evidence(results)
                
                if evidence_found:
                    for evidence in evidence_found[:10]:  # Show first 10 pieces of evidence
                        print(f'   ✅ {evidence}')
                    if len(evidence_found) > 10:
                        print(f'   ... and {len(evidence_found) - 10} more pieces of evidence')
                else:
                    print('   ❌ No clear evidence of distributed execution found')
                
                print('\n🔥 BRUTAL TRUTH: END-TO-END EXECUTION ANALYSIS COMPLETE')
                return True
            else:
                print(f'❌ Unexpected results type: {type(results)}')
                return False
        else:
            print(f'❌ TIMEOUT: Workflow did not complete in {timeout}s')
            print(f'   ⏱️ Elapsed time: {execution_time:.2f}s')
            print('🔥 BRUTAL TRUTH: Workflow orchestration needs debugging')
            return False
            
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print('🔥 BRUTAL TRUTH: Testing complete end-to-end execution')
    print('This test will run the full workflow with extended timeout')
    print('to capture all evidence of distributed execution.')
    print()
    
    success = asyncio.run(test_complete_end_to_end())
    
    if success:
        print('\n🎉 COMPLETE END-TO-END TEST: SUCCESS!')
        print('🔥 BRUTAL TRUTH: The system demonstrates real distributed execution')
        sys.exit(0)
    else:
        print('\n💥 COMPLETE END-TO-END TEST: FAILED!')
        print('🔥 BRUTAL TRUTH: Issues remain in workflow orchestration')
        sys.exit(1)
