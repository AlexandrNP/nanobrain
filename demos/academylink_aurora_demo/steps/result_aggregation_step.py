#!/usr/bin/env python3
"""
BRUTAL TRUTH: Result Aggregation Step - LOCAL EXECUTION ONLY

This step runs locally (no Parsl) and aggregates results from the heavy computation step.
It should have ZERO knowledge that the previous step ran on Parsl.
"""

import asyncio
import time
import statistics
from typing import Dict, Any, Optional, List
from nanobrain.core.step import Step


class ResultAggregationStep(Step):
    """
    BRUTAL TRUTH: Local result aggregation step
    
    This step:
    1. Runs on LOCAL executor (no Parsl)
    2. Has NO knowledge that previous step ran on Parsl
    3. Aggregates computation results into final summary
    4. Demonstrates clean separation of concerns
    """
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize ResultAggregationStep from configuration - FRAMEWORK COMPLIANT"""
        # Call parent initialization first
        super()._init_from_config(config, component_config, dependencies)

        # Extract aggregation_method from config
        self.aggregation_method = getattr(config, 'aggregation_method', 'summary')

        self.nb_logger.info(f"🔧 ResultAggregationStep initialized (LOCAL execution)")
        self.nb_logger.info(f"   Aggregation method: {self.aggregation_method}")
    
    async def process(self, input_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """
        BRUTAL TRUTH: Aggregate computation results locally

        This method:
        1. Gets computation results from input_data parameter
        2. Performs local aggregation and analysis
        3. Returns final results
        4. Has NO knowledge that input came from Parsl execution
        """
        try:
            self.nb_logger.info("🚀 STARTING LOCAL RESULT AGGREGATION")
            self.nb_logger.info("=" * 50)

            # Use input_data parameter provided by framework
            self.nb_logger.info(f"📥 Received computation data: {type(input_data)}")

            # Extract computation results from AcademyLink (from Aurora HPC)
            computation_data = input_data.get('computation_data', input_data)

            if not isinstance(computation_data, dict) or 'computed_sequences' not in computation_data:
                self.nb_logger.error(f"❌ Invalid Aurora computation data format: {computation_data}")
                self.nb_logger.error(f"❌ Expected 'computed_sequences' key, got: {list(computation_data.keys()) if isinstance(computation_data, dict) else type(computation_data)}")
                return None

            computed_sequences = computation_data['computed_sequences']
            computation_metadata = computation_data.get('computation_metadata', {})
            node_information = computation_data.get('node_information', {})

            self.nb_logger.info(f"📊 Aggregating {len(computed_sequences)} computed sequences")
            self.nb_logger.info(f"📋 Computation metadata: {computation_metadata}")

            # Display detailed node information from Aurora workers
            if node_information:
                self.nb_logger.info("🖥️ AURORA NODE INFORMATION:")
                nodes_used = node_information.get('nodes_utilized', [])
                total_nodes = node_information.get('total_nodes', 0)
                self.nb_logger.info(f"   📡 Nodes utilized: {total_nodes} ({', '.join(nodes_used)})")

                perf_summary = node_information.get('performance_summary', {})
                if perf_summary:
                    self.nb_logger.info(f"   🚀 Avg performance factor: {perf_summary.get('avg_performance_factor', 0):.3f}")
                    self.nb_logger.info(f"   📊 Avg utilization: {perf_summary.get('avg_utilization', 0):.1%}")
                    self.nb_logger.info(f"   🔥 Total GPUs: {perf_summary.get('total_gpus', 0)}")
                    self.nb_logger.info(f"   💾 Total memory: {perf_summary.get('total_memory_gb', 0)} GB")
                    self.nb_logger.info(f"   🧠 Total CPU cores: {perf_summary.get('total_cpu_cores', 0)}")

                rack_dist = node_information.get('rack_distribution', {})
                if rack_dist:
                    self.nb_logger.info("   🏗️ Rack distribution:")
                    for rack, rack_nodes in rack_dist.items():
                        self.nb_logger.info(f"      {rack}: {', '.join(rack_nodes)}")
            else:
                self.nb_logger.info("⚠️ No node information received from Aurora workers")
            
            # Perform aggregation analysis
            aggregation_start = time.time()
            
            # Extract metrics for analysis
            complexity_scores = [seq.get('complexity_score', 0) for seq in computed_sequences]
            computation_times = [seq.get('computation_time', 0) for seq in computed_sequences]
            sequence_lengths = [seq.get('length', 0) for seq in computed_sequences]
            
            # Calculate statistics
            aggregated_results = {
                'summary_statistics': {
                    'total_sequences': len(computed_sequences),
                    'complexity_scores': {
                        'mean': statistics.mean(complexity_scores) if complexity_scores else 0,
                        'median': statistics.median(complexity_scores) if complexity_scores else 0,
                        'min': min(complexity_scores) if complexity_scores else 0,
                        'max': max(complexity_scores) if complexity_scores else 0,
                        'stdev': statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0
                    },
                    'computation_times': {
                        'mean': statistics.mean(computation_times) if computation_times else 0,
                        'total': sum(computation_times),
                        'min': min(computation_times) if computation_times else 0,
                        'max': max(computation_times) if computation_times else 0
                    },
                    'sequence_lengths': {
                        'mean': statistics.mean(sequence_lengths) if sequence_lengths else 0,
                        'min': min(sequence_lengths) if sequence_lengths else 0,
                        'max': max(sequence_lengths) if sequence_lengths else 0
                    }
                },
                'top_sequences': [],
                'aggregation_metadata': {
                    'aggregation_method': self.aggregation_method,
                    'aggregation_time': time.time(),
                    'executor_type': 'LOCAL',
                    'input_computation_metadata': computation_metadata
                }
            }
            
            # Find top sequences by complexity score
            sorted_sequences = sorted(computed_sequences, 
                                    key=lambda x: x.get('complexity_score', 0), 
                                    reverse=True)
            
            top_count = min(3, len(sorted_sequences))
            for i in range(top_count):
                seq = sorted_sequences[i]
                top_seq = {
                    'rank': i + 1,
                    'id': seq['id'],
                    'complexity_score': seq.get('complexity_score', 0),
                    'length': seq.get('length', 0),
                    'computation_time': seq.get('computation_time', 0),
                    'heavy_result': seq.get('heavy_result', '')
                }
                aggregated_results['top_sequences'].append(top_seq)
                self.nb_logger.info(f"   🏆 Top {i+1}: {seq['id']} (score: {seq.get('complexity_score', 0):.2f})")

            # Finalize aggregation
            aggregation_end = time.time()
            aggregated_results['aggregation_metadata']['aggregation_duration'] = aggregation_end - aggregation_start

            # Include comprehensive node information in final results
            if node_information:
                aggregated_results['aurora_node_information'] = node_information
                aggregated_results['aggregation_metadata']['aurora_nodes_used'] = node_information.get('total_nodes', 0)
                aggregated_results['aggregation_metadata']['aurora_node_ids'] = node_information.get('nodes_utilized', [])

            # Return final results with correct key mapping for output data units
            # The framework expects keys that match output data unit names
            result = {
                'final_results': aggregated_results  # This matches the output data unit name
            }

            self.nb_logger.info("✅ LOCAL AGGREGATION COMPLETE")
            self.nb_logger.info(f"   📊 Processed: {aggregated_results['summary_statistics']['total_sequences']} sequences")
            self.nb_logger.info(f"   📈 Mean complexity: {aggregated_results['summary_statistics']['complexity_scores']['mean']:.2f}")
            self.nb_logger.info(f"   ⏱️  Aggregation time: {aggregated_results['aggregation_metadata']['aggregation_duration']:.3f}s")
            self.nb_logger.info("   🎯 Final results ready!")
            self.nb_logger.info("=" * 50)
            return result

        except Exception as e:
            self.nb_logger.error(f"❌ Result aggregation failed: {e}")
            import traceback
            self.nb_logger.error(f"Traceback: {traceback.format_exc()}")
            return None
