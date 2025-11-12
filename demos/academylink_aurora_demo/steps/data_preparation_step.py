#!/usr/bin/env python3
"""
BRUTAL TRUTH: Data Preparation Step - LOCAL EXECUTION ONLY

This step runs locally (no Parsl) and prepares data for the heavy computation step.
It should have ZERO knowledge that the next step will run on Parsl.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from nanobrain.core.step import Step


class DataPreparationStep(Step):
    """
    BRUTAL TRUTH: Local data preparation step
    
    This step:
    1. Runs on LOCAL executor (no Parsl)
    2. Has NO knowledge of downstream execution environment
    3. Simply processes data and outputs to data unit
    4. Demonstrates clean separation of concerns
    """
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize DataPreparationStep from configuration - FRAMEWORK COMPLIANT"""
        # Call parent initialization first
        super()._init_from_config(config, component_config, dependencies)

        # Extract processing_delay from config
        self.processing_delay = getattr(config, 'processing_delay', 1.0)

        self.nb_logger.info(f"🔧 DataPreparationStep initialized (LOCAL execution)")
        self.nb_logger.info(f"   Processing delay: {self.processing_delay}s")
    
    async def process(self, input_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """
        BRUTAL TRUTH: Process raw input data locally

        This method:
        1. Gets data from input_data parameter (framework provides this)
        2. Performs local processing (simulation)
        3. Returns prepared data (framework handles output data units)
        4. Has NO knowledge of downstream Parsl execution
        """
        try:
            self.nb_logger.info("🚀 STARTING LOCAL DATA PREPARATION")
            self.nb_logger.info("=" * 50)

            # Use input_data parameter provided by framework
            self.nb_logger.info(f"📥 Received input data: {type(input_data)}")

            # Extract raw_input from input_data dict (framework provides this)
            raw_data = input_data.get('raw_input') if isinstance(input_data, dict) else input_data

            if isinstance(raw_data, dict) and 'sequences' in raw_data:
                sequences = raw_data['sequences']
                self.nb_logger.info(f"📊 Processing {len(sequences)} sequences locally")
            else:
                self.nb_logger.warning(f"⚠️ Unexpected input data format: {raw_data}")
                sequences = []

            # REAL local data preparation - NO SIMULATION
            self.nb_logger.info(f"⚡ Performing REAL local data preparation")
            
            # Prepare data for heavy computation
            prepared_data = {
                'prepared_sequences': [],
                'metadata': {
                    'preparation_time': time.time(),
                    'preparation_method': 'local_processing',
                    'executor_type': 'LOCAL',
                    'total_sequences': len(sequences)
                }
            }
            
            # Process each sequence
            for i, seq in enumerate(sequences):
                prepared_seq = {
                    'id': seq.get('id', f'seq_{i}'),
                    'sequence': seq.get('sequence', ''),
                    'length': len(seq.get('sequence', '')),
                    'prepared_at': time.time(),
                    'preparation_index': i
                }
                prepared_data['prepared_sequences'].append(prepared_seq)
                self.nb_logger.info(f"   ✅ Prepared sequence {i+1}/{len(sequences)}: {prepared_seq['id']}")

            # Return prepared data with correct key mapping for output data units
            # The framework expects keys that match output data unit names
            result = {
                'prepared_data': prepared_data  # This matches the output data unit name
            }

            self.nb_logger.info("✅ LOCAL PREPARATION COMPLETE")
            self.nb_logger.info(f"   📤 Output: {len(prepared_data['prepared_sequences'])} prepared sequences")
            self.nb_logger.info("   🎯 Ready for downstream processing (execution environment unknown)")
            self.nb_logger.info("=" * 50)
            return result

        except Exception as e:
            self.nb_logger.error(f"❌ Data preparation failed: {e}")
            import traceback
            self.nb_logger.error(f"Traceback: {traceback.format_exc()}")
            return None
