"""
Parallel Agent Step Implementation

Specialized parallel processing for NanoBrain agents.

This module provides agent-specific parallel processing capabilities:
- ParallelAgentStep: Generic parallel processing for any agent type
- Configurable agent pools and load balancing
- Agent-specific error handling and recovery
- Performance monitoring for agent operations
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from nanobrain.core.agent import Agent
from .parallel_step import (
    ParallelStep, 
    ParallelProcessingConfig,
    ProcessingRequest,
    ProcessingResponse,
    LoadBalancingStrategy
)


@dataclass
class AgentRequest(ProcessingRequest):
    """Request for agent processing."""
    input_data: Any = None
    agent_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'input_data': self.input_data,
            'agent_context': self.agent_context
        })
        return base_dict


@dataclass 
class AgentResponse(ProcessingResponse):
    """Response from agent processing."""
    output_data: Any = None
    agent_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'output_data': self.output_data,
            'agent_metadata': self.agent_metadata
        })
        return base_dict


class ParallelAgentConfig(ParallelProcessingConfig):
    """Configuration for parallel agent processing."""
    agent_initialization_timeout: float = 10.0
    agent_shutdown_timeout: float = 5.0
    enable_agent_health_checks: bool = True
    health_check_interval: float = 30.0
    auto_restart_failed_agents: bool = True
    agent_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ParallelAgentStep(ParallelStep[AgentRequest, AgentResponse, Agent]):
    """
    Parallel processing step specialized for NanoBrain agents.
    
    This class provides:
    - Agent pool management with health monitoring
    - Agent-specific load balancing strategies
    - Automatic agent recovery and restart
    - Agent performance tracking and optimization
    """
    
    def __init__(self, config: ParallelAgentConfig, agents: List[Agent], **kwargs):
        super().__init__(config, agents, **kwargs)
        self.agent_config = config
        self.agent_health_status = {i: {'healthy': True, 'last_check': datetime.now(), 'consecutive_failures': 0}
                                   for i in range(len(agents))}
        
        # Agent-specific performance tracking
        self.agent_performance = {i: {'total_requests': 0, 'successful_requests': 0, 'avg_response_time': 0.0}
                                 for i in range(len(agents))}
        
        # Health check task
        self.health_check_task = None
        
        self.logger.info(f"Initialized ParallelAgentStep with {len(agents)} agents",
                        agent_count=len(agents),
                        load_balancing_strategy=config.load_balancing_strategy.value)
    
    async def initialize(self) -> None:
        """Initialize the parallel agent step and all agents."""
        await super().initialize()
        
        # Initialize all agents
        self.logger.info("Initializing agents in parallel")
        initialization_tasks = []
        
        for i, agent in enumerate(self.processor_pool.processors):
            task = self._initialize_agent_with_timeout(agent, i)
            initialization_tasks.append(task)
        
        # Wait for all agents to initialize
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Check initialization results
        successful_agents = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to initialize agent {i}: {result}")
                self.agent_health_status[i]['healthy'] = False
                self.agent_health_status[i]['consecutive_failures'] += 1
            else:
                successful_agents += 1
        
        if successful_agents == 0:
            raise RuntimeError("Failed to initialize any agents")
        
        self.logger.info(f"Successfully initialized {successful_agents}/{len(self.processor_pool.processors)} agents")
        
        # Start health check task if enabled
        if self.agent_config.enable_agent_health_checks:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def shutdown(self) -> None:
        """Shutdown the parallel agent step and all agents."""
        # Stop health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all agents
        self.logger.info("Shutting down agents in parallel")
        shutdown_tasks = []
        
        for i, agent in enumerate(self.processor_pool.processors):
            task = self._shutdown_agent_with_timeout(agent, i)
            shutdown_tasks.append(task)
        
        # Wait for all agents to shutdown
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        await super().shutdown()
    
    async def _initialize_agent_with_timeout(self, agent: Agent, agent_index: int) -> None:
        """Initialize an agent with timeout."""
        try:
            await asyncio.wait_for(
                agent.initialize(),
                timeout=self.agent_config.agent_initialization_timeout
            )
            self.logger.debug(f"Agent {agent_index} initialized successfully")
        except asyncio.TimeoutError:
            raise RuntimeError(f"Agent {agent_index} initialization timed out")
        except Exception as e:
            raise RuntimeError(f"Agent {agent_index} initialization failed: {e}")
    
    async def _shutdown_agent_with_timeout(self, agent: Agent, agent_index: int) -> None:
        """Shutdown an agent with timeout."""
        try:
            await asyncio.wait_for(
                agent.shutdown(),
                timeout=self.agent_config.agent_shutdown_timeout
            )
            self.logger.debug(f"Agent {agent_index} shutdown successfully")
        except asyncio.TimeoutError:
            self.logger.warning(f"Agent {agent_index} shutdown timed out")
        except Exception as e:
            self.logger.error(f"Agent {agent_index} shutdown failed: {e}")
    
    async def _health_check_loop(self) -> None:
        """Continuous health check loop for agents."""
        while True:
            try:
                await asyncio.sleep(self.agent_config.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all agents."""
        self.logger.debug("Performing agent health checks")
        
        health_check_tasks = []
        for i, agent in enumerate(self.processor_pool.processors):
            task = self._check_agent_health(agent, i)
            health_check_tasks.append(task)
        
        results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
        
        # Process health check results
        healthy_agents = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Health check failed for agent {i}: {result}")
                self._mark_agent_unhealthy(i)
            elif result:
                self._mark_agent_healthy(i)
                healthy_agents += 1
            else:
                self._mark_agent_unhealthy(i)
        
        self.logger.debug(f"Health check complete: {healthy_agents}/{len(self.processor_pool.processors)} agents healthy")
    
    async def _check_agent_health(self, agent: Agent, agent_index: int) -> bool:
        """Check the health of a specific agent."""
        try:
            # Simple health check - verify agent is initialized and responsive
            if not hasattr(agent, 'is_initialized') or not agent.is_initialized:
                return False
            
            # Could add more sophisticated health checks here
            # For example, sending a test request or checking resource usage
            
            return True
        except Exception as e:
            self.logger.debug(f"Agent {agent_index} health check failed: {e}")
            return False
    
    def _mark_agent_healthy(self, agent_index: int) -> None:
        """Mark an agent as healthy."""
        health_status = self.agent_health_status[agent_index]
        health_status['healthy'] = True
        health_status['last_check'] = datetime.now()
        health_status['consecutive_failures'] = 0
        
        # Re-enable in circuit breaker if it was disabled
        self.processor_pool.circuit_breakers[agent_index]['is_open'] = False
    
    def _mark_agent_unhealthy(self, agent_index: int) -> None:
        """Mark an agent as unhealthy."""
        health_status = self.agent_health_status[agent_index]
        health_status['healthy'] = False
        health_status['last_check'] = datetime.now()
        health_status['consecutive_failures'] += 1
        
        # Open circuit breaker for unhealthy agent
        self.processor_pool.circuit_breakers[agent_index]['is_open'] = True
        
        # Attempt restart if configured
        if (self.agent_config.auto_restart_failed_agents and 
            health_status['consecutive_failures'] >= 3):
            asyncio.create_task(self._restart_agent(agent_index))
    
    async def _restart_agent(self, agent_index: int) -> None:
        """Restart a failed agent."""
        self.logger.info(f"Attempting to restart agent {agent_index}")
        
        try:
            agent = self.processor_pool.processors[agent_index]
            
            # Shutdown the agent
            await self._shutdown_agent_with_timeout(agent, agent_index)
            
            # Reinitialize the agent
            await self._initialize_agent_with_timeout(agent, agent_index)
            
            # Mark as healthy if restart successful
            self._mark_agent_healthy(agent_index)
            
            self.logger.info(f"Successfully restarted agent {agent_index}")
            
        except Exception as e:
            self.logger.error(f"Failed to restart agent {agent_index}: {e}")
    
    async def _extract_requests(self, inputs: Dict[str, Any]) -> List[AgentRequest]:
        """Extract agent requests from input data."""
        requests = []
        
        # Handle different input formats
        if 'requests' in inputs:
            # Batch of requests
            request_data = inputs['requests']
            if not isinstance(request_data, list):
                request_data = [request_data]
            
            for data in request_data:
                request = self._create_agent_request(data)
                if request:
                    requests.append(request)
        
        elif 'input_data' in inputs or 'user_input' in inputs:
            # Single request
            input_data = inputs.get('input_data') or inputs.get('user_input')
            request = self._create_agent_request(input_data)
            if request:
                requests.append(request)
        
        else:
            # Try to create request from entire input
            request = self._create_agent_request(inputs)
            if request:
                requests.append(request)
        
        return requests
    
    def _create_agent_request(self, data: Any) -> Optional[AgentRequest]:
        """Create an AgentRequest from input data."""
        try:
            if isinstance(data, dict):
                # Extract nested user input if present
                if 'user_input' in data:
                    input_data = data['user_input']
                    if isinstance(input_data, dict):
                        input_data = input_data.get('user_input', str(input_data))
                else:
                    input_data = data
                
                return AgentRequest(
                    input_data=input_data,
                    agent_context=data.get('context', {}),
                    priority=data.get('priority', 1),
                    timeout=data.get('timeout')
                )
            else:
                # Simple data type
                return AgentRequest(input_data=data)
                
        except Exception as e:
            self.logger.error(f"Failed to create agent request from data: {e}")
            return None
    
    async def _execute_processor(self, processor: Agent, request: AgentRequest, processor_index: int) -> AgentResponse:
        """Execute an agent with a request."""
        start_time = time.time()
        
        try:
            # Check if agent is healthy
            if not self.agent_health_status[processor_index]['healthy']:
                raise RuntimeError(f"Agent {processor_index} is marked as unhealthy")
            
            # Process the request with the agent
            if hasattr(processor, 'process'):
                # Use agent's process method
                result = await processor.process(request.input_data)
            elif hasattr(processor, 'execute'):
                # Use agent's execute method
                result = await processor.execute(request.input_data)
            else:
                raise RuntimeError(f"Agent {processor_index} has no process or execute method")
            
            processing_time = time.time() - start_time
            
            # Update agent performance metrics
            self._update_agent_performance(processor_index, processing_time, True)
            
            return AgentResponse(
                request_id=request.id,
                output_data=result,
                processing_time=processing_time,
                processor_id=f"agent_{processor_index}",
                success=True,
                agent_metadata={
                    'agent_type': type(processor).__name__,
                    'agent_index': processor_index
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update agent performance metrics for failure
            self._update_agent_performance(processor_index, processing_time, False)
            
            # Mark agent as potentially unhealthy
            self.agent_health_status[processor_index]['consecutive_failures'] += 1
            
            raise e
    
    async def _create_error_response(self, request: AgentRequest, error_message: str, 
                                   processing_time: float = 0.0, processor_id: str = "") -> AgentResponse:
        """Create an error response for a failed agent request."""
        return AgentResponse(
            request_id=request.id,
            output_data=None,
            processing_time=processing_time,
            processor_id=processor_id,
            success=False,
            error=error_message,
            agent_metadata={'error_type': 'processing_failure'}
        )
    
    def _update_agent_performance(self, agent_index: int, processing_time: float, success: bool):
        """Update performance metrics for a specific agent."""
        perf = self.agent_performance[agent_index]
        perf['total_requests'] += 1
        
        if success:
            perf['successful_requests'] += 1
            
            # Update average response time
            if perf['avg_response_time'] == 0:
                perf['avg_response_time'] = processing_time
            else:
                # Exponential moving average
                alpha = 0.1
                perf['avg_response_time'] = (alpha * processing_time + 
                                           (1 - alpha) * perf['avg_response_time'])
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        return {
            'agent_count': len(self.processor_pool.processors),
            'healthy_agents': sum(1 for status in self.agent_health_status.values() if status['healthy']),
            'agent_health_status': self.agent_health_status,
            'agent_performance': self.agent_performance,
            'processor_pool_stats': self.processor_pool.get_stats()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including agent-specific metrics."""
        base_stats = super().get_performance_stats()
        agent_stats = self.get_agent_stats()
        
        return {**base_stats, 'agent_stats': agent_stats} 