"""
Tracked Query Enhancement Step
================================

Query enhancement step with journey tracking.
Simple extension that logs query enhancement to journey logger.
"""

import time
from typing import Any, Dict, Optional

import sys
from pathlib import Path

# Add the pbs_parallel_rag directory to path for local imports
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from pbs_query_enhancement_step import PBSQueryEnhancementStep
from journey_logging.journey_logger import QueryJourneyLogger
from nanobrain.core.shared_resource import get_worker_id


class TrackedQueryEnhancementStep(PBSQueryEnhancementStep):
    """
    Query enhancement step with journey tracking.
    
    Extends PBSQueryEnhancementStep to log:
    - Original query
    - Enhanced query
    - Worker ID
    - Processing time
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
        Process query with journey tracking.
        
        Tracks:
        - Original query
        - Enhanced query
        - Worker ID
        - Instance ID
        - Processing time
        """
        # Extract query from args/kwargs
        original_query = None
        query_id = None
        
        # Try to get query from kwargs
        if 'query' in kwargs:
            original_query = kwargs['query']
        elif len(args) > 0 and isinstance(args[0], str):
            original_query = args[0]
        
        # Try to get query_id from kwargs
        if 'query_id' in kwargs:
            query_id = kwargs['query_id']
        
        # Start journey if logger available and we have a query
        if self.journey_logger and original_query:
            if not query_id:
                # Generate query ID if not provided
                query_id = f"q_{int(time.time() * 1000)}"
                kwargs['query_id'] = query_id
            
            # Start journey
            self.journey_logger.start_journey(query_id, original_query)
            
            if self.nb_logger:
                self.nb_logger.info(f"📝 Started journey tracking for query: {query_id}")
        
        # Process with timing
        start_time = time.time()
        result = await super().process(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        # Update journey with enhancement results
        if self.journey_logger and query_id:
            # Extract enhanced query from result
            enhanced_query = None
            if isinstance(result, dict):
                enhanced_query = result.get('enhanced_query') or result.get('query')
            elif isinstance(result, str):
                enhanced_query = result

            if enhanced_query:
                # Get worker ID
                worker_id = get_worker_id()
                if isinstance(result, dict) and '_worker_id' in result:
                    worker_id = result['_worker_id']

                # Update journey
                self.journey_logger.update_enhancement(
                    query_id=query_id,
                    enhanced_query=enhanced_query,
                    worker_id=worker_id,
                    instance_id=id(self),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration
                )

                if self.nb_logger:
                    self.nb_logger.info(f"✅ Updated journey with enhancement (worker: {worker_id}, time: {duration:.2f}s)")
        
        return result

