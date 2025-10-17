"""
Tracked Response Generation Step
=================================

Response generation step with journey tracking.
Simple extension that logs final response to journey logger.
"""

import time
from typing import Any, Dict, Optional

from nanobrain.library.workflows.rag.steps import ResponseEnhancementStep
import sys
from pathlib import Path

# Add the pbs_parallel_rag directory to path for local imports
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from journey_logging.journey_logger import QueryJourneyLogger
from nanobrain.core.shared_resource import get_worker_id


class TrackedResponseGenerationStep(ResponseEnhancementStep):
    """
    Response generation step with journey tracking.

    Extends ResponseEnhancementStep to log:
    - Final response
    - Worker ID
    - Processing time
    - Complete journey
    """
    
    def __init__(self, config: Dict[str, Any], journey_logger: Optional[QueryJourneyLogger] = None):
        """
        Initialize tracked step.
        
        Args:
            config: Step configuration
            journey_logger: Logger for tracking query journeys
        """
        super().__init__(config)
        self.journey_logger = journey_logger
    
    async def process(self, *args, **kwargs):
        """
        Process response generation with journey tracking.
        
        Tracks:
        - Final response
        - Worker ID
        - Instance ID
        - Processing time
        - Completes and saves journey
        """
        # Get query_id from kwargs
        query_id = kwargs.get('query_id')
        
        # Process with timing
        start_time = time.time()
        result = await super().process(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        # Update journey with generation results
        if self.journey_logger and query_id:
            # Extract response from result
            final_response = None
            if isinstance(result, dict):
                final_response = result.get('response') or result.get('answer') or result.get('text')
            elif isinstance(result, str):
                final_response = result

            if final_response:
                # Get worker ID
                worker_id = get_worker_id()
                if isinstance(result, dict) and '_worker_id' in result:
                    worker_id = result['_worker_id']

                # Update journey
                self.journey_logger.update_generation(
                    query_id=query_id,
                    final_response=final_response,
                    worker_id=worker_id,
                    instance_id=id(self),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration
                )

                # Complete journey
                self.journey_logger.complete_journey(query_id)

                # Save journey to files
                self.journey_logger.save_journey(query_id)

                if self.nb_logger:
                    self.nb_logger.info(f"✅ Updated journey with generation (worker: {worker_id}, time: {duration:.2f}s)")
                    self.nb_logger.info(f"💾 Journey saved for query: {query_id}")
        
        return result

