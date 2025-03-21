from typing import Any, List, Optional, Dict, Union
from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase
from src.TriggerBase import TriggerBase
from src.enums import ComponentState
from pydantic import Field
import asyncio
import time

class DataStorageBase(Step):
    """
    Base class for data storage operations that respond to triggers.
    
    Biological analogy: Memory system with retrieval mechanism.
    Justification: Like how memory systems in the brain store information and
    retrieve it in response to specific cues, this class stores data and
    produces output in response to trigger conditions.
    """
    # Fields for Pydantic/BaseTool validation
    input: Any = Field(default=None, exclude=True)
    last_query: Any = Field(default=None, exclude=True)
    last_response: Any = Field(default=None, exclude=True)
    processing_history: List = Field(default_factory=list, exclude=True)
    max_history_size: int = Field(default=10, exclude=True)
    
    def __init__(self, 
                 executor: ExecutorBase,
                 input_unit: DataUnitBase,
                 output_unit: DataUnitBase,
                 trigger: TriggerBase,
                 **kwargs):
        """
        Initialize the DataStorageBase.
        
        Args:
            executor: The executor responsible for running this step
            input_unit: The data unit to read input from
            output_unit: The data unit to write output to
            trigger: The trigger that activates this storage operation
            **kwargs: Additional keyword arguments
        """
        # Initialize with proper name for BaseTool
        name = kwargs.get('name', self.__class__.__name__)
        description = kwargs.get('description', self.__doc__ or f"Data storage operation")
        
        super().__init__(
            executor=executor, 
            name=name,
            description=description,
            **kwargs
        )
        
        self.input = input_unit
        self.output = output_unit
        self.trigger = trigger
        
        # Configure the trigger to activate this step
        self.trigger.runnable = self
        
        # Storage-specific attributes
        self.last_query = None
        self.last_response = None
        self.processing_history = []
        self.max_history_size = kwargs.get('max_history_size', 10)
        
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process the input data and produce output.
        
        Biological analogy: Memory retrieval and encoding.
        Justification: Like how memory systems retrieve stored information
        based on cues and encode new information, this method processes
        input queries and produces appropriate responses.
        
        Args:
            inputs: List of input data (typically a single query)
            
        Returns:
            The processed output data
        """
        # Extract the query from inputs
        query = inputs[0] if inputs else None
        self.last_query = query
        
        # If no query provided, return None
        if query is None:
            return None
        
        # Process the query (to be implemented by subclasses)
        response = await self._process_query(query)
        self.last_response = response
        
        # Store the result in the output unit
        self.output.set(response)
        
        # Record this processing in history
        self._update_history(query, response)
        
        return response
    
    async def _process_query(self, query: Any) -> Any:
        """
        Process a query and produce a response.
        
        This method should be overridden by subclasses to implement
        specific storage and retrieval logic.
        
        Args:
            query: The query to process
            
        Returns:
            The response to the query
        """
        # Default implementation just returns the query
        return query
    
    def _update_history(self, query: Any, response: Any) -> None:
        """
        Update the processing history.
        
        Args:
            query: The processed query
            response: The generated response
        """
        # Add to history
        self.processing_history.append({
            'query': query,
            'response': response,
            'timestamp': time.time()
        })
        
        # Trim history if it exceeds max size
        if len(self.processing_history) > self.max_history_size:
            self.processing_history = self.processing_history[-self.max_history_size:]
    
    async def start_monitoring(self):
        """
        Start monitoring for trigger conditions.
        
        Biological analogy: Attention mechanism.
        Justification: Like how attention mechanisms in the brain continuously
        monitor for relevant stimuli, this method continuously monitors for
        trigger conditions.
        """
        # Set the state to active (monitoring)
        self._state = ComponentState.ACTIVE
        
        # Start the trigger's monitoring loop
        await self.trigger.monitor()
    
    async def stop_monitoring(self):
        """
        Stop monitoring for trigger conditions.
        
        Biological analogy: Attention shift.
        Justification: Like how attention can shift away from previously
        monitored stimuli, this method stops monitoring for trigger conditions.
        """
        # Set the state to inactive
        self._state = ComponentState.INACTIVE
        
        # Stop the trigger's monitoring loop
        if hasattr(self.trigger, 'stop_monitoring'):
            await self.trigger.stop_monitoring()
    
    def get_history(self) -> List[Dict]:
        """
        Get the processing history.
        
        Returns:
            List of processing history entries
        """
        return self.processing_history
    
    def clear_history(self):
        """
        Clear the processing history.
        
        Biological analogy: Memory clearance.
        Justification: Like how certain memory systems can be cleared or reset,
        this method clears the processing history.
        """
        self.processing_history = []
        
    def get_last_interaction(self) -> Dict:
        """
        Get the last query-response interaction.
        
        Returns:
            Dictionary containing the last query and response
        """
        if not self.last_query or not self.last_response:
            return {}
            
        return {
            'query': self.last_query,
            'response': self.last_response
        }
    
    # BaseTool required methods
    
    def _run(self, *args: Any, run_manager: Optional[Any] = None) -> Any:
        """
        Use the data storage as a tool synchronously.
        
        Biological analogy: Conscious memory retrieval.
        Justification: Like how we can consciously retrieve stored memories,
        this method retrieves data from storage in response to specific queries.
        """
        loop = asyncio.get_event_loop()
        # Call the process method with the input arguments
        if len(args) == 1 and isinstance(args[0], list):
            # If input is a list, pass it directly to process
            return loop.run_until_complete(self.process(args[0]))
        else:
            # Otherwise, package the arguments as a list
            return loop.run_until_complete(self.process(list(args)))
    
    async def _arun(self, *args: Any, run_manager: Optional[Any] = None) -> Any:
        """
        Use the data storage as a tool asynchronously.
        
        Biological analogy: Automatic memory association.
        Justification: Like how memories can be automatically retrieved through
        associative processes, this method asynchronously retrieves data.
        """
        # Call the process method with the input arguments
        if len(args) == 1 and isinstance(args[0], list):
            # If input is a list, pass it directly to process
            return await self.process(args[0])
        else:
            # Otherwise, package the arguments as a list
            return await self.process(list(args)) 