"""
Distributed Processing Step

A NanoBrain step that handles distributed processing of multiple messages
across multiple agents using Parsl for parallel execution.

This step follows NanoBrain architecture principles:
- Proper separation of concerns
- Data unit integration
- Logging and monitoring
- Error handling and recovery
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from nanobrain.core.logging_system import get_logger, OperationType
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig, DataUnitType


class DistributedProcessingStep:
    """
    Step for distributed processing of multiple messages across agents.
    
    This step:
    - Takes a list of messages from input data unit
    - Distributes them across available agents
    - Coordinates parallel execution via Parsl
    - Collects and aggregates results
    - Stores results in output data unit
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the distributed processing step."""
        self.config = config
        self.name = config.get('name', 'distributed_processing_step')
        self.logger = get_logger(f"step_{self.name}", category="steps")
        
        # Step state
        self.is_initialized = False
        self.agents = []
        self.parsl_executor = None
        
        # Data units for input/output
        self.input_data_unit = None
        self.output_data_unit = None
        
    async def initialize(self, agents: List[Any], parsl_executor: Any) -> None:
        """Initialize the step with agents and executor."""
        if self.is_initialized:
            return
            
        self.logger.info(f"Initializing {self.name}")
        
        self.agents = agents
        self.parsl_executor = parsl_executor
        
        # Setup data units
        await self._setup_data_units()
        
        self.is_initialized = True
        self.logger.info(f"{self.name} initialized successfully")
        
    async def _setup_data_units(self) -> None:
        """Setup input and output data units."""
        # Input data unit for messages
        self.input_data_unit = DataUnitMemory(
            DataUnitConfig(
                name=f"{self.name}_input",
                type=DataUnitType.MEMORY,
                description="Input messages for distributed processing"
            )
        )
        await self.input_data_unit.initialize()
        
        # Output data unit for results
        self.output_data_unit = DataUnitMemory(
            DataUnitConfig(
                name=f"{self.name}_output",
                type=DataUnitType.MEMORY,
                description="Results from distributed processing"
            )
        )
        await self.output_data_unit.initialize()
        
    async def execute(self, messages: List[str]) -> Dict[str, Any]:
        """
        Execute distributed processing of messages.
        
        Args:
            messages: List of messages to process
            
        Returns:
            Dict containing processing results and metadata
        """
        if not self.is_initialized:
            raise RuntimeError(f"{self.name} not initialized")
            
        self.logger.info(f"Executing distributed processing for {len(messages)} messages")
        
        # Store input in data unit
        await self.input_data_unit.set({
            'messages': messages,
            'timestamp': datetime.now().isoformat(),
            'agent_count': len(self.agents)
        })
        
        try:
            # Distribute messages across agents
            results = await self._distribute_messages(messages)
            
            # Store results in output data unit
            await self.output_data_unit.set(results)
            
            self.logger.info(f"Distributed processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in distributed processing: {e}")
            error_result = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            await self.output_data_unit.set(error_result)
            return error_result
            
    async def _distribute_messages(self, messages: List[str]) -> Dict[str, Any]:
        """
        Distribute messages across agents using round-robin allocation.
        
        Args:
            messages: List of messages to distribute
            
        Returns:
            Dict containing all results and metadata
        """
        if not self.agents:
            raise ValueError("No agents available for processing")
            
        self.logger.info(f"Distributing {len(messages)} messages across {len(self.agents)} agents")
        
        # Create agent-message pairs using round-robin distribution
        agent_message_pairs = []
        for i, message in enumerate(messages):
            agent_index = i % len(self.agents)
            agent = self.agents[agent_index]
            agent_message_pairs.append((agent, message, i))
            
        # Log distribution plan
        for agent, message, index in agent_message_pairs:
            self.logger.debug(f"Message {index} -> Agent {agent.config.name}: {message[:50]}...")
            
        # Execute processing in parallel
        tasks = []
        for agent, message, index in agent_message_pairs:
            task = asyncio.create_task(
                self._process_single_message(agent, message, index)
            )
            tasks.append(task)
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and aggregate results
        return self._aggregate_results(results, agent_message_pairs)
        
    async def _process_single_message(self, agent: Any, message: str, index: int) -> Dict[str, Any]:
        """
        Process a single message with an agent.
        
        Args:
            agent: The agent to process the message
            message: The message to process
            index: Message index for tracking
            
        Returns:
            Dict containing the result
        """
        start_time = datetime.now()
        
        # Log processing start
        if hasattr(agent, 'nb_logger'):
            agent.nb_logger.info(
                f"Agent {agent.config.name} starting distributed message processing",
                message_index=index,
                input_message=message[:100] + "..." if len(message) > 100 else message
            )
            
        try:
            # Process the message
            response = await agent.process(message)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log success
            if hasattr(agent, 'nb_logger'):
                agent.nb_logger.info(
                    f"Agent {agent.config.name} completed distributed message processing",
                    message_index=index,
                    response_length=len(response),
                    duration_seconds=duration,
                    success=True
                )
                
            return {
                'success': True,
                'message': message,
                'response': response,
                'agent': agent.config.name,
                'index': index,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log failure
            if hasattr(agent, 'nb_logger'):
                agent.nb_logger.error(
                    f"Agent {agent.config.name} failed distributed message processing",
                    message_index=index,
                    error=str(e),
                    duration_seconds=duration,
                    success=False
                )
                
            return {
                'success': False,
                'message': message,
                'error': str(e),
                'agent': agent.config.name,
                'index': index,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            }
            
    def _aggregate_results(self, results: List[Any], agent_message_pairs: List[tuple]) -> Dict[str, Any]:
        """
        Aggregate results from all agent processing tasks.
        
        Args:
            results: List of results from asyncio.gather
            agent_message_pairs: Original agent-message pairs for context
            
        Returns:
            Aggregated results dictionary
        """
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions from asyncio.gather
                agent, message, index = agent_message_pairs[i]
                failed_results.append({
                    'success': False,
                    'message': message,
                    'error': str(result),
                    'agent': agent.config.name,
                    'index': index,
                    'timestamp': datetime.now().isoformat()
                })
            elif result.get('success', False):
                successful_results.append(result)
            else:
                failed_results.append(result)
                
        # Calculate statistics
        total_messages = len(agent_message_pairs)
        success_count = len(successful_results)
        failure_count = len(failed_results)
        
        # Log summary
        self.logger.info(
            f"Distributed processing summary: {success_count}/{total_messages} successful, "
            f"{failure_count} failed"
        )
        
        return {
            'success': success_count > 0,
            'total_messages': total_messages,
            'successful_count': success_count,
            'failed_count': failure_count,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'timestamp': datetime.now().isoformat(),
            'method': 'distributed_processing'
        }
        
    async def get_input_data(self) -> Optional[Dict[str, Any]]:
        """Get the current input data."""
        if self.input_data_unit:
            return await self.input_data_unit.get()
        return None
        
    async def get_output_data(self) -> Optional[Dict[str, Any]]:
        """Get the current output data."""
        if self.output_data_unit:
            return await self.output_data_unit.get()
        return None
        
    async def shutdown(self) -> None:
        """Shutdown the step and cleanup resources."""
        self.logger.info(f"Shutting down {self.name}")
        self.is_initialized = False 