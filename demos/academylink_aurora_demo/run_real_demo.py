#!/usr/bin/env python3
"""
Real AcademyLink Aurora Demo

This demo showcases REAL integration between Nanobrain and Academy
for distributed Aurora HPC computation using real AcademyLink with deployed agents.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.workflow import Workflow

# Import real Academy agents
from agents.aurora_computation_agent import AuroraComputationAgent
from agents.aurora_results_agent import AuroraResultsAgent

# Import Academy framework
try:
    from academy.manager import Manager
    from academy.exchange.local import LocalExchangeFactory
    ACADEMY_AVAILABLE = True
except ImportError:
    print("❌ Academy framework not available")
    ACADEMY_AVAILABLE = False
    sys.exit(1)


async def run_real_academy_demo():
    """Run the real Academy demo with deployed agents"""
    print('🔥 BRUTAL TRUTH: REAL ACADEMY INTEGRATION DEMO')
    print('=' * 70)
    
    if not ACADEMY_AVAILABLE:
        print("❌ Academy framework not available - cannot run real demo")
        return
    
    try:
        # Create Academy manager
        print('🚀 Creating Academy manager...')
        exchange_factory = LocalExchangeFactory()
        
        async with await Manager.from_exchange_factory(exchange_factory) as academy_manager:
            print('✅ Academy Manager initialized')
            
            # Deploy real Academy agents
            print('🚀 Deploying Academy agents...')
            
            # Create and launch Aurora computation agent
            aurora_comp_agent = AuroraComputationAgent()
            aurora_comp_handle = await academy_manager.launch(aurora_comp_agent)
            print(f'   ✅ Aurora Computation Agent: {aurora_comp_handle}')
            
            # Create and launch Aurora results agent
            aurora_results_agent = AuroraResultsAgent()
            aurora_results_handle = await academy_manager.launch(aurora_results_agent)
            print(f'   ✅ Aurora Results Agent: {aurora_results_handle}')
            
            print('✅ All Academy agents deployed successfully')
            
            # Load workflow with real AcademyLink
            print('\n🔗 Loading workflow with real AcademyLink...')
            workflow = Workflow.from_config('config/mixed_execution_workflow_aurora.yml')

            # Initialize workflow (Academy agents are already deployed and available)
            await workflow.initialize()
            print('✅ Workflow initialized with real Academy integration')
            
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
                    'source': 'real_academy_demo',
                    'timestamp': time.time(),
                    'demo_type': 'real_distributed_execution'
                }
            }
            
            print(f'📊 Test data prepared: {len(test_data["sequences"])} sequences')
            
            # Execute workflow
            print('\n🚀 Executing real distributed workflow...')
            print('   🔄 Data preparation -> Academy Aurora computation -> Result aggregation')
            
            # Set input data using workflow-level input data units
            input_unit = workflow.input_data_units['raw_input']
            await input_unit.set(test_data)
            print('✅ Input data set - workflow executing with real Academy agents...')

            # Wait for completion with timeout
            timeout_seconds = 300  # 5 minutes for real execution
            start_time = time.time()

            # Get output data unit from workflow level
            output_unit = workflow.output_data_units['final_results']

            while True:
                # Check if final results are available
                try:
                    if output_unit.has_data():
                        final_results = await output_unit.get()
                    if final_results:
                        elapsed = time.time() - start_time
                        print(f'\n🎯 REAL DISTRIBUTED EXECUTION COMPLETED!')
                        print(f'   ⏱️ Total execution time: {elapsed:.2f}s')
                        
                        # Display results
                        print(f'\n📊 FINAL RESULTS:')
                        if isinstance(final_results, dict):
                            for key, value in final_results.items():
                                if key == 'top_sequences' and isinstance(value, list):
                                    print(f'   🏆 Top sequences: {len(value)} results')
                                    for i, seq in enumerate(value[:3], 1):
                                        if isinstance(seq, dict):
                                            seq_id = seq.get('id', f'seq_{i}')
                                            score = seq.get('score', 'N/A')
                                            print(f'      {i}. {seq_id}: {score}')
                                elif key == 'statistics':
                                    print(f'   📈 Statistics: {value}')
                                else:
                                    print(f'   {key}: {value}')
                        
                        print(f'\n✅ REAL ACADEMY INTEGRATION DEMO COMPLETED SUCCESSFULLY!')
                        return
                        
                except Exception as e:
                    pass  # Results not ready yet
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f'\n❌ TIMEOUT: Workflow did not complete in {timeout_seconds}s')
                    print(f'   ⏱️ Elapsed time: {elapsed:.2f}s')
                    break
                
                # Wait and show progress
                if int(elapsed) % 10 == 0:
                    print(f'   ⏳ Still executing... {int(elapsed)}s elapsed')
                    print(f'      🔄 Real Academy agents processing on Aurora HPC')
                
                await asyncio.sleep(1)
            
    except Exception as e:
        print(f'❌ Demo failed: {e}')
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    await run_real_academy_demo()


if __name__ == "__main__":
    asyncio.run(main())
