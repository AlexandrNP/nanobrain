#!/usr/bin/env python3
"""
NanoBrain Chat Workflow Demo with Parsl Executor

A comprehensive demonstration of the NanoBrain framework featuring:
- Parsl executor for distributed/parallel execution
- Multiple parallel conversational agents
- CLI input/output interface with load balancing
- Performance monitoring and comparison
- Resource management and scaling
- Comprehensive logging with execution tracking

Architecture:
CLI Input → Load Balancer → Multiple Parallel Agents (via Parsl) → Response Aggregator → CLI Output

This demo showcases:
1. Parallel processing of chat requests across multiple agents
2. Distributed execution using Parsl for HPC/cloud environments
3. Load balancing and resource management
4. Performance monitoring and metrics collection
5. Fault tolerance and error handling in distributed systems
"""

import asyncio
import sys
import os
import threading
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import uuid

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

# Import NanoBrain components
from core.data_unit import DataUnitMemory, DataUnitConfig
from core.trigger import DataUpdatedTrigger, TriggerConfig
from core.link import DirectLink, LinkConfig
from core.step import Step, StepConfig
from core.agent import ConversationalAgent, AgentConfig
from core.executor import ParslExecutor, ExecutorConfig, ExecutorType
from config.component_factory import ComponentFactory, get_factory

# Import logging system
from core.logging_system import NanoBrainLogger, get_logger, set_debug_mode, OperationType

# Import global configuration
try:
    sys.path.insert(0, os.path.join(current_dir, '..', 'config'))
    from config_manager import get_config_manager, get_api_key, get_provider_config, get_logging_config, should_log_to_file, should_log_to_console
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import configuration manager: {e}")
    print("   API keys will need to be set via environment variables")
    CONFIG_AVAILABLE = False


@dataclass
class ChatRequest:
    """Represents a chat request with metadata."""
    id: str
    message: str
    timestamp: datetime
    user_id: str = "default_user"
    priority: int = 1
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class ChatResponse:
    """Represents a chat response with metadata."""
    request_id: str
    response: str
    agent_id: str
    processing_time: float
    timestamp: datetime
    tokens_used: int = 0
    error: Optional[str] = None


class ParslLogManager:
    """
    Enhanced log manager for Parsl-based workflows.
    
    Provides distributed logging capabilities and performance tracking
    across multiple parallel execution contexts.
    """
    
    def __init__(self, base_log_dir: Optional[str] = None):
        # Get global logging configuration
        try:
            self.logging_config = get_logging_config()
            self.should_log_to_file = should_log_to_file()
            self.should_log_to_console = should_log_to_console()
            
            file_config = self.logging_config.get('file', {})
            if base_log_dir is None:
                base_log_dir = file_config.get('base_directory', 'logs')
            self.use_session_directories = file_config.get('use_session_directories', True)
            
        except ImportError:
            self.logging_config = {}
            self.should_log_to_file = True
            self.should_log_to_console = True
            if base_log_dir is None:
                base_log_dir = "logs"
            self.use_session_directories = True
        
        self.base_log_dir = Path(base_log_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_session_directories:
            self.session_dir = self.base_log_dir / f"parsl_session_{self.session_id}"
        else:
            self.session_dir = self.base_log_dir
            
        self.loggers = {}
        self.performance_metrics = {}
        
        if self.should_log_to_file:
            self._setup_log_directories()
        
    def _setup_log_directories(self):
        """Create organized log directory structure for Parsl execution."""
        directories = [
            self.session_dir,
            self.session_dir / "parsl",
            self.session_dir / "agents", 
            self.session_dir / "performance",
            self.session_dir / "distributed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        if self.should_log_to_console:
            print(f"🚀 Created Parsl log session directory: {self.session_dir}")
        
    def get_logger(self, name: str, category: str = "parsl", 
                   debug_mode: bool = True) -> NanoBrainLogger:
        """Get or create a logger for Parsl components."""
        logger_key = f"{category}_{name}"
        
        if logger_key not in self.loggers:
            log_file = None
            if self.should_log_to_file:
                log_file = self.session_dir / category / f"{name}.log"
            
            logger = NanoBrainLogger(
                name=f"{category}.{name}",
                log_file=log_file,
                debug_mode=debug_mode
            )
            
            self.loggers[logger_key] = logger
            
            if self.should_log_to_console or self.should_log_to_file:
                logger.info(f"Parsl logger initialized for {name}", 
                           category=category, 
                           session_id=self.session_id,
                           parsl_enabled=True)
        
        return self.loggers[logger_key]
    
    def track_performance(self, metric_name: str, value: float, context: Dict[str, Any] = None):
        """Track performance metrics for Parsl execution."""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                metric_values = [v['value'] for v in values]
                summary[metric_name] = {
                    'count': len(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'avg': sum(metric_values) / len(metric_values),
                    'total': sum(metric_values)
                }
        
        return summary


class ParallelConversationalAgentStep(Step):
    """
    Enhanced conversational agent step that leverages Parsl for parallel execution.
    
    This step can process multiple chat requests in parallel using distributed resources.
    """
    
    def __init__(self, config: StepConfig, agents: List[ConversationalAgent], 
                 log_manager: ParslLogManager, agent_id: str = None):
        super().__init__(config)
        self.agents = agents
        self.log_manager = log_manager
        self.agent_id = agent_id or f"parallel_agent_{uuid.uuid4().hex[:8]}"
        self.logger = log_manager.get_logger(self.agent_id, "agents")
        self.request_count = 0
        self.total_processing_time = 0.0
        
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process chat requests in parallel using available agents.
        
        Args:
            inputs: Dictionary containing chat requests or batch of requests
            
        Returns:
            Dictionary containing responses and metadata
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Extract chat requests (support both single and batch)
            if 'requests' in inputs:
                # Batch processing
                requests = inputs['requests']
                if not isinstance(requests, list):
                    requests = [requests]
            else:
                # Single request
                request_data = inputs.get('message', inputs.get('user_input', ''))
                requests = [ChatRequest(
                    id=str(uuid.uuid4()),
                    message=request_data,
                    timestamp=datetime.now()
                )]
            
            self.logger.info(f"Processing {len(requests)} chat requests in parallel",
                           agent_id=self.agent_id,
                           request_count=len(requests),
                           available_agents=len(self.agents))
            
            # Process requests in parallel using available agents
            responses = await self._process_requests_parallel(requests)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Track performance metrics
            self.log_manager.track_performance('request_processing_time', processing_time, {
                'agent_id': self.agent_id,
                'request_count': len(requests),
                'response_count': len(responses)
            })
            
            self.logger.info(f"Completed parallel processing",
                           agent_id=self.agent_id,
                           processing_time=processing_time,
                           requests_processed=len(requests),
                           responses_generated=len(responses),
                           avg_processing_time=self.total_processing_time / self.request_count)
            
            return {
                'responses': responses,
                'metadata': {
                    'agent_id': self.agent_id,
                    'processing_time': processing_time,
                    'request_count': len(requests),
                    'response_count': len(responses),
                    'parallel_execution': True
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in parallel chat processing: {str(e)}",
                            agent_id=self.agent_id,
                            error_type=type(e).__name__,
                            processing_time=processing_time)
            
            return {
                'responses': [],
                'error': str(e),
                'metadata': {
                    'agent_id': self.agent_id,
                    'processing_time': processing_time,
                    'error': True
                }
            }
    
    async def _process_requests_parallel(self, requests: List[ChatRequest]) -> List[ChatResponse]:
        """Process multiple chat requests in parallel using available agents."""
        
        # Create tasks for parallel processing
        tasks = []
        for i, request in enumerate(requests):
            # Round-robin agent assignment
            agent = self.agents[i % len(self.agents)]
            task = self._process_single_request(request, agent, f"agent_{i % len(self.agents)}")
            tasks.append(task)
        
        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Request {requests[i].id} failed: {str(response)}")
                valid_responses.append(ChatResponse(
                    request_id=requests[i].id,
                    response="Sorry, I encountered an error processing your request.",
                    agent_id=self.agent_id,
                    processing_time=0.0,
                    timestamp=datetime.now(),
                    error=str(response)
                ))
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _process_single_request(self, request: ChatRequest, 
                                    agent: ConversationalAgent, 
                                    agent_label: str) -> ChatResponse:
        """Process a single chat request using a specific agent."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing request {request.id} with {agent_label}",
                            request_id=request.id,
                            agent_label=agent_label,
                            message_length=len(request.message))
            
            # Process with agent - ConversationalAgent.process() expects a string
            result = await agent.process(request.message)
            
            processing_time = time.time() - start_time
            
            # ConversationalAgent.process() returns a string directly
            response_text = result if isinstance(result, str) else str(result)
            tokens_used = 0  # Token counting would need to be implemented in the agent
            
            return ChatResponse(
                request_id=request.id,
                response=response_text,
                agent_id=f"{self.agent_id}_{agent_label}",
                processing_time=processing_time,
                timestamp=datetime.now(),
                tokens_used=tokens_used
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing request {request.id}: {str(e)}",
                            request_id=request.id,
                            agent_label=agent_label,
                            error_type=type(e).__name__)
            
            return ChatResponse(
                request_id=request.id,
                response=f"Error: {str(e)}",
                agent_id=f"{self.agent_id}_{agent_label}",
                processing_time=processing_time,
                timestamp=datetime.now(),
                error=str(e)
            )


class LoadBalancedCLIInterface:
    """
    Enhanced CLI interface with load balancing for parallel agent processing.
    
    Features:
    - Request queuing and batching
    - Load balancing across multiple agents
    - Performance monitoring
    - Real-time statistics display
    """
    
    def __init__(self, input_data_unit: DataUnitMemory, output_data_unit: DataUnitMemory, 
                 log_manager: ParslLogManager):
        self.input_data_unit = input_data_unit
        self.output_data_unit = output_data_unit
        self.log_manager = log_manager
        self.logger = log_manager.get_logger("cli_interface", "parsl")
        
        self.running = False
        self.input_thread = None
        self.request_queue = asyncio.Queue()
        self.response_stats = {
            'total_requests': 0,
            'total_responses': 0,
            'total_processing_time': 0.0,
            'error_count': 0
        }
        
    async def start(self):
        """Start the load-balanced CLI interface."""
        self.running = True
        
        print("🚀 NanoBrain Parsl Chat Workflow Demo")
        print("=" * 50)
        print("Features:")
        print("  • Parallel processing with Parsl executor")
        print("  • Multiple conversational agents")
        print("  • Load balancing and performance monitoring")
        print("  • Distributed execution capabilities")
        print()
        print("Commands:")
        print("  /help     - Show this help")
        print("  /stats    - Show performance statistics")
        print("  /batch N  - Send N test messages for load testing")
        print("  /quit     - Exit the demo")
        print("=" * 50)
        print()
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        # Start output monitoring
        asyncio.create_task(self._monitor_output())
        
        self.logger.info("Load-balanced CLI interface started",
                        parsl_enabled=True,
                        load_balancing=True)
    
    async def stop(self):
        """Stop the CLI interface."""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        print("\n👋 Parsl Chat Workflow Demo stopped")
        self._show_final_stats()
        
        self.logger.info("CLI interface stopped", 
                        final_stats=self.response_stats)
    
    def _input_loop(self):
        """Input loop running in separate thread."""
        while self.running:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # Queue regular message for processing
                    asyncio.run_coroutine_threadsafe(
                        self._queue_message(user_input),
                        asyncio.get_event_loop()
                    )
                    
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Input error: {e}")
    
    async def _queue_message(self, message: str):
        """Queue a message for processing."""
        request = ChatRequest(
            id=str(uuid.uuid4()),
            message=message,
            timestamp=datetime.now()
        )
        
        await self.request_queue.put(request)
        
        # Send to input data unit
        await self.input_data_unit.set_data({
            'user_input': message,
            'request_id': request.id,
            'timestamp': request.timestamp.isoformat(),
            'batch_processing': False
        })
        
        self.response_stats['total_requests'] += 1
    
    def _handle_command(self, command: str):
        """Handle special commands."""
        if command == '/help':
            self._show_help()
        elif command == '/stats':
            self._show_stats()
        elif command.startswith('/batch '):
            try:
                count = int(command.split()[1])
                asyncio.run_coroutine_threadsafe(
                    self._send_batch_requests(count),
                    asyncio.get_event_loop()
                )
            except (ValueError, IndexError):
                print("Usage: /batch <number>")
        elif command == '/quit':
            self.running = False
        else:
            print(f"Unknown command: {command}")
    
    async def _send_batch_requests(self, count: int):
        """Send batch requests for load testing."""
        print(f"🔄 Sending {count} batch requests for load testing...")
        
        requests = []
        for i in range(count):
            request = ChatRequest(
                id=str(uuid.uuid4()),
                message=f"Test message {i+1}: What can you tell me about parallel processing?",
                timestamp=datetime.now()
            )
            requests.append(request)
        
        # Send batch to input data unit
        await self.input_data_unit.set_data({
            'requests': [
                {
                    'user_input': req.message,
                    'request_id': req.id,
                    'timestamp': req.timestamp.isoformat()
                } for req in requests
            ],
            'batch_processing': True,
            'batch_size': count
        })
        
        self.response_stats['total_requests'] += count
        print(f"✅ Sent {count} requests for parallel processing")
    
    async def _monitor_output(self):
        """Monitor output data unit for responses."""
        while self.running:
            try:
                # Check for new output data
                output_data = await self.output_data_unit.get_data()
                
                if output_data:
                    await self._handle_response(output_data)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Output monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_response(self, response_data: Dict[str, Any]):
        """Handle response from agents."""
        try:
            if 'responses' in response_data:
                # Batch response
                responses = response_data['responses']
                metadata = response_data.get('metadata', {})
                
                print(f"\n📦 Batch Response ({len(responses)} responses):")
                print(f"   Processing time: {metadata.get('processing_time', 0):.2f}s")
                print(f"   Agent: {metadata.get('agent_id', 'unknown')}")
                
                for i, resp in enumerate(responses[:3]):  # Show first 3
                    if isinstance(resp, dict):
                        response_text = resp.get('response', str(resp))
                    else:
                        response_text = str(resp)
                    print(f"   Response {i+1}: {response_text[:100]}...")
                
                if len(responses) > 3:
                    print(f"   ... and {len(responses) - 3} more responses")
                
                self.response_stats['total_responses'] += len(responses)
                self.response_stats['total_processing_time'] += metadata.get('processing_time', 0)
                
            else:
                # Single response
                response_text = response_data.get('agent_response', 
                                                response_data.get('response', str(response_data)))
                metadata = response_data.get('metadata', {})
                
                print(f"\nBot: {response_text}")
                
                if metadata.get('processing_time'):
                    print(f"     (processed in {metadata['processing_time']:.2f}s by {metadata.get('agent_id', 'agent')})")
                
                self.response_stats['total_responses'] += 1
                self.response_stats['total_processing_time'] += metadata.get('processing_time', 0)
            
            # Check for errors
            if response_data.get('error'):
                self.response_stats['error_count'] += 1
                print(f"⚠️  Error: {response_data['error']}")
            
        except Exception as e:
            self.logger.error(f"Error handling response: {e}")
            print(f"⚠️  Error displaying response: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n📖 Parsl Chat Workflow Demo Help")
        print("=" * 40)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /stats    - Show performance statistics")
        print("  /batch N  - Send N test messages for load testing")
        print("  /quit     - Exit the demo")
        print()
        print("Features:")
        print("  • Messages are processed in parallel using Parsl")
        print("  • Multiple agents handle requests simultaneously")
        print("  • Load balancing distributes work efficiently")
        print("  • Performance metrics are tracked in real-time")
        print("=" * 40)
    
    def _show_stats(self):
        """Show current performance statistics."""
        stats = self.response_stats
        avg_time = (stats['total_processing_time'] / max(stats['total_responses'], 1))
        
        print("\n📊 Performance Statistics")
        print("=" * 30)
        print(f"Total Requests:    {stats['total_requests']}")
        print(f"Total Responses:   {stats['total_responses']}")
        print(f"Error Count:       {stats['error_count']}")
        print(f"Total Proc. Time:  {stats['total_processing_time']:.2f}s")
        print(f"Avg Response Time: {avg_time:.2f}s")
        
        # Show performance metrics from log manager
        perf_summary = self.log_manager.get_performance_summary()
        if perf_summary:
            print("\nDetailed Metrics:")
            for metric, data in perf_summary.items():
                print(f"  {metric}:")
                print(f"    Count: {data['count']}")
                print(f"    Avg:   {data['avg']:.3f}")
                print(f"    Min:   {data['min']:.3f}")
                print(f"    Max:   {data['max']:.3f}")
        
        print("=" * 30)
    
    def _show_final_stats(self):
        """Show final statistics on shutdown."""
        print("\n📈 Final Performance Report")
        print("=" * 40)
        self._show_stats()
        
        # Additional Parsl-specific metrics
        print("\nParsl Execution Summary:")
        print(f"  Parallel processing enabled: ✅")
        print(f"  Distributed execution: ✅")
        print(f"  Load balancing: ✅")
        print("=" * 40)


class ParslChatWorkflow:
    """
    Main workflow class that orchestrates the Parsl-based chat system.
    
    This workflow demonstrates:
    - Parsl executor configuration and initialization
    - Multiple parallel conversational agents
    - Load balancing and resource management
    - Performance monitoring and metrics collection
    """
    
    def __init__(self):
        self.logger = get_logger("parsl_workflow")
        
        # Core components
        self.parsl_executor = None
        self.agents = []
        
        # Configuration
        self.num_agents = 3  # Number of parallel agents
        self.api_key = None
        
    def _setup_api_keys(self):
        """Setup API keys for LLM providers."""
        try:
            if CONFIG_AVAILABLE:
                config_manager = get_config_manager()
                self.api_key = get_api_key('openai')
                
                if not self.api_key:
                    self.logger.warning("No OpenAI API key found in configuration")
            
            # Fallback to environment variables
            if not self.api_key:
                self.api_key = os.getenv('OPENAI_API_KEY')
            
            if not self.api_key:
                print("⚠️  Warning: No OpenAI API key found!")
                print("   Set OPENAI_API_KEY environment variable or configure in config manager")
                print("   The demo will use mock responses for demonstration")
                return False
            
            # Set environment variable for agents
            os.environ['OPENAI_API_KEY'] = self.api_key
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up API keys: {e}")
            return False
    
    async def setup(self):
        """Setup the Parsl chat workflow."""
        self.logger.info("Setting up Parsl Chat Workflow")
        
        # Setup API keys
        api_available = self._setup_api_keys()
        
        # Create Parsl executor with configuration
        parsl_config = {
            'executor_type': ExecutorType.PARSL,
            'max_workers': 8,  # Allow up to 8 parallel workers
            'parsl_config': {
                'executors': [{
                    'class': 'parsl.executors.HighThroughputExecutor',
                    'label': 'htex_local',
                    'max_workers': 8,
                    'cores_per_worker': 1
                }]
            }
        }
        
        executor_config = ExecutorConfig(**parsl_config)
        self.parsl_executor = ParslExecutor(config=executor_config)
        
        try:
            await self.parsl_executor.initialize()
            self.logger.info("Parsl executor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Parsl executor: {e}")
            print(f"⚠️  Parsl initialization failed: {e}")
            print("   Falling back to local execution for demo")
            
            # Fallback to local executor
            from core.executor import LocalExecutor
            self.parsl_executor = LocalExecutor()
            await self.parsl_executor.initialize()
        
        # Create multiple conversational agents for parallel processing
        self.agents = []
        for i in range(self.num_agents):
            agent_config = AgentConfig(
                name=f"parsl_agent_{i}",
                description=f"Parallel conversational agent {i+1} for distributed processing",
                model="gpt-3.5-turbo" if api_available else "mock",
                temperature=0.7,
                max_tokens=500,
                system_prompt=f"""You are Agent {i+1} in a parallel processing system. 
You're part of a distributed chat workflow using Parsl for high-performance computing.
Be helpful, concise, and mention that you're processing requests in parallel when appropriate.
Your responses should be informative and engaging."""
            )
            
            agent = ConversationalAgent(
                config=agent_config,
                executor=self.parsl_executor
            )
            
            # Explicitly initialize the agent to ensure LLM client is set up
            await agent.initialize()
            
            self.agents.append(agent)
            self.logger.info(f"Created agent {i+1}/{self.num_agents} (LLM client: {agent.llm_client is not None})")
        
        self.logger.info("Parsl Chat Workflow setup complete",
                        num_agents=len(self.agents),
                        executor_type=type(self.parsl_executor).__name__,
                        api_available=api_available)
    
    async def process_message(self, message: str) -> str:
        """Process a message using parallel agents."""
        try:
            # Create chat request
            request = ChatRequest(
                id=str(uuid.uuid4()),
                message=message,
                timestamp=datetime.now()
            )
            
            # Select agent (round-robin for simplicity)
            agent_index = hash(request.id) % len(self.agents)
            selected_agent = self.agents[agent_index]
            
            self.logger.info(f"Processing message with agent {agent_index}")
            
            # Process with selected agent - ConversationalAgent.process() expects a string
            start_time = time.time()
            result = await selected_agent.process(message)
            processing_time = time.time() - start_time
            
            # ConversationalAgent.process() returns a string directly
            response_text = result if isinstance(result, str) else str(result)
            
            self.logger.info(f"Message processed in {processing_time:.2f}s")
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def run_interactive(self):
        """Run interactive chat session."""
        print("🚀 NanoBrain Parsl Chat Workflow Demo")
        print("=" * 50)
        print("Features:")
        print("  • Parallel processing with Parsl executor")
        print("  • Multiple conversational agents")
        print("  • Distributed execution capabilities")
        print()
        print("Commands:")
        print("  /help     - Show this help")
        print("  /quit     - Exit the demo")
        print("=" * 50)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/quit':
                    break
                elif user_input == '/help':
                    print("\n📖 Parsl Chat Workflow Demo Help")
                    print("This demo uses Parsl executor for parallel processing")
                    print("Your messages are processed by multiple agents in parallel")
                    print("Type /quit to exit")
                    continue
                
                # Process message
                response = await self.process_message(user_input)
                print(f"Bot: {response}")
                
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n👋 Parsl Chat Workflow Demo stopped")
    
    async def shutdown(self):
        """Shutdown the workflow and cleanup resources."""
        self.logger.info("Shutting down Parsl Chat Workflow")
        
        try:
            # Shutdown Parsl executor
            if self.parsl_executor:
                await self.parsl_executor.shutdown()
                self.logger.info("Parsl executor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point for the Parsl chat workflow demo."""
    
    # Enable debug mode for comprehensive logging
    set_debug_mode(True)
    
    print("🚀 Initializing NanoBrain Parsl Chat Workflow Demo...")
    
    try:
        # Create and setup workflow
        workflow = ParslChatWorkflow()
        await workflow.setup()
        
        # Run the interactive session
        await workflow.run_interactive()
        
        # Shutdown
        await workflow.shutdown()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Demo completed.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 