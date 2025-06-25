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
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig
from nanobrain.core.link import DirectLink, LinkConfig
from nanobrain.core.step import Step, StepConfig
from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.executor import ParslExecutor, ExecutorConfig, ExecutorType
from nanobrain.core.config.component_factory import ComponentFactory

# Import logging system
from nanobrain.core.logging_system import (
    NanoBrainLogger, get_logger, set_debug_mode, OperationType,
    get_system_log_manager, log_component_lifecycle, register_component,
    log_workflow_event, create_session_summary
)

# Import global configuration
try:
    from nanobrain.core.config import get_config_manager, get_api_key, get_provider_config, get_logging_config, should_log_to_file, should_log_to_console
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import configuration manager: {e}")
    print("   API keys will need to be set via environment variables")
    CONFIG_AVAILABLE = False
    # Provide fallback functions
    def get_logging_config():
        return {}
    def should_log_to_file():
        return True
    def should_log_to_console():
        return True
    def get_config_manager():
        return None
    def get_api_key(provider):
        return os.getenv(f'{provider.upper()}_API_KEY')
    def get_provider_config(provider):
        return None


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
    
    def create_session_summary(self):
        """Create a summary of the logging session."""
        summary_file = self.session_dir / "session_summary.json"
        
        summary = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "log_directory": str(self.session_dir),
            "loggers_created": list(self.loggers.keys()),
            "log_files": []
        }
        
        # Collect all log files
        for log_file in self.session_dir.rglob("*.log"):
            if log_file.is_file():
                summary["log_files"].append({
                    "name": log_file.name,
                    "path": str(log_file.relative_to(self.session_dir)),
                    "size_bytes": log_file.stat().st_size if log_file.exists() else 0
                })
        
        # Write summary
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
    
    def cleanup_old_logs(self, keep_days: int = 7):
        """Clean up old log sessions."""
        if not self.base_log_dir.exists():
            return
            
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        
        for session_dir in self.base_log_dir.glob("parsl_session_*"):
            if session_dir.is_dir():
                try:
                    if session_dir.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(session_dir)
                        print(f"🗑️  Cleaned up old log session: {session_dir.name}")
                except Exception as e:
                    print(f"⚠️  Could not clean up {session_dir}: {e}")


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
                # Single request - handle nested dictionary structure
                user_input_data = inputs.get('user_input', '')
                
                # If user_input is a dictionary, extract the actual message
                if isinstance(user_input_data, dict):
                    request_data = user_input_data.get('user_input', '')
                else:
                    request_data = user_input_data
                
                # Ensure request_data is a string
                if not isinstance(request_data, str):
                    request_data = str(request_data) if request_data else ''
                
                if not request_data or request_data.strip() == '':
                    self.logger.warning("Empty user input received")
                    return {'responses': [], 'metadata': {'agent_id': self.agent_id, 'error': 'empty_input'}}
                
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
                    'parallel_execution': True,
                    'batch_processing': len(requests) > 1 or inputs.get('batch_processing', False)
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
        self.request_queue = None  # Will be created when event loop is available
        self.event_loop = None  # Store reference to the event loop
        self.waiting_for_response = False  # Flag to coordinate input/output
        self.response_stats = {
            'total_requests': 0,
            'total_responses': 0,
            'total_processing_time': 0.0,
            'error_count': 0
        }
        
    async def start(self):
        """Start the load-balanced CLI interface."""
        self.running = True
        
        # Store the current event loop for use in the input thread
        self.event_loop = asyncio.get_running_loop()
        
        # Create the request queue now that we have an event loop
        self.request_queue = asyncio.Queue()
        
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
        
        # Start input processing task
        asyncio.create_task(self._process_input_queue())
        
        self.logger.info("Load-balanced CLI interface started",
                        parsl_enabled=True,
                        load_balancing=True)
    
    async def stop(self):
        """Stop the CLI interface."""
        self.running = False
        
        self.logger.info("Stopping CLI interface")
        
        # Wait for input thread to finish
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
                # Wait if we're expecting a response to avoid prompt interference
                while self.waiting_for_response and self.running:
                    import time
                    time.sleep(0.1)
                
                if not self.running:
                    break
                    
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Queue all input for async processing (including commands)
                if self.event_loop and not self.event_loop.is_closed():
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self._queue_message(user_input),
                            self.event_loop
                        )
                        # Wait for the coroutine to complete with a timeout
                        future.result(timeout=1.0)
                    except Exception as e:
                        print(f"❌ Error queuing message: {e}")
                        self.waiting_for_response = False
                        break
                else:
                    print("❌ Event loop not available")
                    self.waiting_for_response = False
                    break
                    
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"Input error: {e}")
                self.waiting_for_response = False
    
    async def _queue_message(self, message: str):
        """Queue a message for processing."""
        await self.request_queue.put(message)
    
    async def _process_input_queue(self):
        """Process input from the queue."""
        try:
            while self.running:
                try:
                    # Wait for input with timeout
                    user_input = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                    
                    if user_input.lower() in ['quit', 'exit', 'bye', '/quit']:
                        print("👋 Goodbye!")
                        self.logger.info("User requested exit")
                        self.running = False
                        break
                        
                    if user_input.lower() in ['help', '/help']:
                        self._show_help()
                        continue
                        
                    if user_input.lower() in ['stats', '/stats']:
                        self._show_stats()
                        continue
                    
                    if user_input.startswith('/batch '):
                        try:
                            count = int(user_input.split()[1])
                            await self._send_batch_requests(count)
                        except (ValueError, IndexError):
                            print("Usage: /batch <number>")
                        continue
                    
                    # Log user input
                    self.logger.info("User input received", 
                                    input_length=len(user_input),
                                    input_preview=user_input[:50] + "..." if len(user_input) > 50 else user_input)
                    
                    # Set waiting flag before sending input
                    self.waiting_for_response = True
                    
                    # Send input to data unit (this will trigger the workflow)
                    await self.input_data_unit.set({
                        'user_input': user_input,
                        'request_id': str(uuid.uuid4()),
                        'timestamp': datetime.now().isoformat(),
                        'batch_processing': False
                    })
                    
                    self.response_stats['total_requests'] += 1
                    
                except asyncio.TimeoutError:
                    # No input received, continue loop
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error processing input: {e}", error_type=type(e).__name__)
            print(f"❌ Error processing input: {e}")
    

    
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
        
        # Set waiting flag for batch processing
        self.waiting_for_response = True
        
        # Send batch to input data unit
        await self.input_data_unit.set({
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
        last_output_time = 0.0
        
        while self.running:
            try:
                # Check if output data unit has been updated
                if hasattr(self.output_data_unit, 'get_metadata'):
                    current_update_time = await self.output_data_unit.get_metadata('last_updated', 0.0)
                    
                    if current_update_time > last_output_time:
                        last_output_time = current_update_time
                        output_data = await self.output_data_unit.get()
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
                responses = response_data['responses']
                metadata = response_data.get('metadata', {})
                
                # Check if this is a single response or actual batch
                if len(responses) == 1 and not metadata.get('batch_processing', False):
                    # Single response - display as normal bot response
                    resp = responses[0]
                    if hasattr(resp, 'response'):
                        response_text = resp.response
                    elif isinstance(resp, dict):
                        response_text = resp.get('response', str(resp))
                    else:
                        response_text = str(resp)
                    
                    # Clean up the response text if it contains error info
                    if response_text.startswith('Error: Error code:'):
                        response_text = "I apologize, but I encountered an error processing your request. Please try again."
                    
                    print(f"\n🤖 Bot: {response_text}")
                    
                    # Show processing info in debug mode only
                    if metadata.get('processing_time') and metadata.get('processing_time') > 1.0:
                        agent_id = metadata.get('agent_id', 'agent')
                        if hasattr(resp, 'agent_id'):
                            agent_id = resp.agent_id
                        print(f"     (processed in {metadata['processing_time']:.2f}s by {agent_id})")
                    
                    # Add a small delay to ensure response is visible before next prompt
                    print()  # Extra newline for spacing
                    
                    # Clear waiting flag to allow next input
                    self.waiting_for_response = False
                    
                    self.response_stats['total_responses'] += 1
                    self.response_stats['total_processing_time'] += metadata.get('processing_time', 0)
                    
                else:
                    # Actual batch response
                    print(f"\n📦 Batch Response ({len(responses)} responses):")
                    print(f"   Processing time: {metadata.get('processing_time', 0):.2f}s")
                    print(f"   Agent: {metadata.get('agent_id', 'unknown')}")
                    
                    for i, resp in enumerate(responses):
                        if hasattr(resp, 'response'):
                            response_text = resp.response
                        elif isinstance(resp, dict):
                            response_text = resp.get('response', str(resp))
                        else:
                            response_text = str(resp)
                        
                        # Show full response for batch (up to reasonable length)
                        if len(response_text) > 200:
                            display_text = response_text[:200] + "..."
                        else:
                            display_text = response_text
                        print(f"   Response {i+1}: {display_text}")
                    
                    # Add spacing after batch response
                    print()
                    
                    # Clear waiting flag to allow next input
                    self.waiting_for_response = False
                    
                    self.response_stats['total_responses'] += len(responses)
                    self.response_stats['total_processing_time'] += metadata.get('processing_time', 0)
                
            else:
                # Legacy single response format
                response_text = response_data.get('agent_response', 
                                                response_data.get('response', str(response_data)))
                metadata = response_data.get('metadata', {})
                
                print(f"\n🤖 Bot: {response_text}")
                
                if metadata.get('processing_time') and metadata.get('processing_time') > 1.0:
                    print(f"     (processed in {metadata['processing_time']:.2f}s by {metadata.get('agent_id', 'agent')})")
                
                # Add spacing after legacy response
                print()
                
                # Clear waiting flag to allow next input
                self.waiting_for_response = False
                
                self.response_stats['total_responses'] += 1
                self.response_stats['total_processing_time'] += metadata.get('processing_time', 0)
            
            # Check for errors
            if response_data.get('error'):
                self.response_stats['error_count'] += 1
                print(f"⚠️  Error: {response_data['error']}")
                # Clear waiting flag on error
                self.waiting_for_response = False
            
        except Exception as e:
            self.logger.error(f"Error handling response: {e}")
            print(f"⚠️  Error displaying response: {e}")
            # Clear waiting flag on exception
            self.waiting_for_response = False
    
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
    Main workflow class that orchestrates the Parsl-based chat system following NanoBrain architecture.
    
    This workflow demonstrates:
    - Complete NanoBrain component integration (data units, triggers, links, steps)
    - Parsl executor for distributed/parallel execution
    - Multiple parallel conversational agents
    - Event-driven processing with triggers
    - Data flow through direct links
    - Performance monitoring and metrics collection
    
    Architecture:
    CLI Input → User Input DataUnit → Agent Input DataUnit → ParallelAgentStep → Agent Output DataUnit → CLI Output
    """
    
    def __init__(self):
        self.factory = ComponentFactory()
        self.components = {}
        self.cli = None
        self.parsl_executor = None
        self.agents = []
        
        # Configuration
        self.num_agents = 3  # Number of parallel agents
        self.api_key = None
        
        # Initialize log manager
        self.log_manager = ParslLogManager()
        self.main_logger = self.log_manager.get_logger("parsl_workflow", "workflows")
        
        # Enable debug mode globally
        set_debug_mode(True)
        
        # Register this workflow component
        register_component("workflows", "parsl_workflow", self, {
            "description": "Parsl-based chat workflow with complete NanoBrain architecture",
            "num_agents": self.num_agents
        })
        
        # Log workflow initialization
        log_workflow_event("parsl_workflow", "initialize", {
            "num_agents": self.num_agents,
            "architecture": "complete_nanobrain"
        })
        
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
        """Set up the complete Parsl chat workflow following NanoBrain architecture."""
        log_workflow_event("parsl_workflow", "setup_start")
        self.main_logger.info("Starting NanoBrain Parsl Chat Workflow setup")
        
        # Only show setup messages if console logging is enabled
        if self.log_manager.should_log_to_console:
            print("🔧 Setting up NanoBrain Parsl Chat Workflow...")
        
        # 0. Setup API keys from global configuration
        if self.log_manager.should_log_to_console:
            print("   Setting up API keys...")
        api_available = self._setup_api_keys()
        
        # 1. Create Parsl executor
        if self.log_manager.should_log_to_console:
            print("   Creating Parsl executor...")
        self.main_logger.info("Creating ParslExecutor")
        
        # Create Parsl executor with configuration
        parsl_config = {
            'executor_type': ExecutorType.PARSL,
            'max_workers': 8,  # Allow up to 8 parallel workers
            'parsl_config': {
                'executors': [{
                    'class': 'parsl.executors.HighThroughputExecutor',
                    'label': 'htex_local',
                    'max_workers_per_node': 8,
                    'cores_per_worker': 1
                }]
            }
        }
        
        executor_config = ExecutorConfig(**parsl_config)
        self.parsl_executor = ParslExecutor(config=executor_config)
        
        # Register executor component
        register_component("executors", "parsl_executor", self.parsl_executor, {
            "type": "ParslExecutor",
            "max_workers": 8
        })
        
        try:
            log_component_lifecycle("executors", "parsl_executor", "initialize")
            await self.parsl_executor.initialize()
            log_component_lifecycle("executors", "parsl_executor", "initialized")
            self.main_logger.info("Parsl executor initialized successfully")
        except Exception as e:
            log_component_lifecycle("executors", "parsl_executor", "failed", {"error": str(e)})
            self.main_logger.error(f"Failed to initialize Parsl executor: {e}")
            if self.log_manager.should_log_to_console:
                print(f"⚠️  Parsl initialization failed: {e}")
                print("   Falling back to local execution for demo")
            
            # Fallback to local executor
            from nanobrain.core.executor import LocalExecutor
            executor_config = ExecutorConfig(
                executor_type="local",
                max_workers=4,
                timeout=30.0
            )
            self.parsl_executor = LocalExecutor(executor_config)
            register_component("executors", "local_executor", self.parsl_executor, {
                "type": "LocalExecutor",
                "fallback": True
            })
            log_component_lifecycle("executors", "local_executor", "initialize")
            await self.parsl_executor.initialize()
            log_component_lifecycle("executors", "local_executor", "initialized")
        
        # 2. Create data units
        if self.log_manager.should_log_to_console:
            print("   Creating data units...")
        self.main_logger.info("Creating data units")
        
        # User input data unit
        user_input_config = DataUnitConfig(
            name="user_input",
            data_type="memory",
            persistent=False,
            cache_size=100
        )
        self.components['user_input_du'] = DataUnitMemory(user_input_config)
        
        # Agent input data unit  
        agent_input_config = DataUnitConfig(
            name="agent_input", 
            data_type="memory",
            persistent=False,
            cache_size=100
        )
        self.components['agent_input_du'] = DataUnitMemory(agent_input_config)
        
        # Agent output data unit
        agent_output_config = DataUnitConfig(
            name="agent_output",
            data_type="memory", 
            persistent=False,
            cache_size=100
        )
        self.components['agent_output_du'] = DataUnitMemory(agent_output_config)
        
        # 3. Create multiple conversational agents for parallel processing
        if self.log_manager.should_log_to_console:
            print("   Creating parallel conversational agents...")
        self.main_logger.info("Creating parallel ConversationalAgents")
        
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
Your responses should be informative and engaging.""",
                auto_initialize=False,
                debug_mode=True,
                enable_logging=True,
                log_conversations=True
            )
            
            agent = ConversationalAgent(
                config=agent_config,
                executor=self.parsl_executor
            )
            
            # Create agent logger
            agent_logger = self.log_manager.get_logger(f"parsl_agent_{i}", "agents")
            agent.nb_logger = agent_logger
            
            # Register agent component
            register_component("agents", f"parsl_agent_{i}", agent, {
                "model": agent_config.model,
                "temperature": agent_config.temperature,
                "max_tokens": agent_config.max_tokens
            })
            
            # Initialize the agent
            log_component_lifecycle("agents", f"parsl_agent_{i}", "initialize")
            await agent.initialize()
            log_component_lifecycle("agents", f"parsl_agent_{i}", "initialized")
            
            self.agents.append(agent)
            self.main_logger.info(f"Created agent {i+1}/{self.num_agents}")
        
        # 4. Create parallel conversational agent step
        if self.log_manager.should_log_to_console:
            print("   Creating parallel agent step...")
        self.main_logger.info("Creating ParallelConversationalAgentStep")
        step_config = StepConfig(
            name="parallel_chat_agent_step",
            description="Parallel conversational agent step for distributed chat processing",
            debug_mode=True
        )
        
        self.components['agent_step'] = ParallelConversationalAgentStep(
            step_config, 
            self.agents,
            self.log_manager
        )
        
        # Set up step with data units
        self.components['agent_step'].register_input_data_unit(
            'user_input', 
            self.components['agent_input_du']
        )
        self.components['agent_step'].register_output_data_unit(
            self.components['agent_output_du']
        )
        
        await self.components['agent_step'].initialize()
        
        # 5. Create triggers
        if self.log_manager.should_log_to_console:
            print("   Creating triggers...")
        self.main_logger.info("Creating triggers")
        
        # Trigger for user input → agent input
        user_trigger_config = TriggerConfig(
            name="user_input_trigger",
            trigger_type="data_updated"
        )
        self.components['user_trigger'] = DataUpdatedTrigger(
            [self.components['user_input_du']], 
            user_trigger_config
        )
        
        user_trigger_logger = self.log_manager.get_logger("user_input_trigger", "triggers")
        self.components['user_trigger'].nb_logger = user_trigger_logger
        
        # Trigger for agent input → agent processing
        agent_trigger_config = TriggerConfig(
            name="agent_input_trigger", 
            trigger_type="data_updated"
        )
        self.components['agent_trigger'] = DataUpdatedTrigger(
            [self.components['agent_input_du']], 
            agent_trigger_config
        )
        
        agent_trigger_logger = self.log_manager.get_logger("agent_input_trigger", "triggers")
        self.components['agent_trigger'].nb_logger = agent_trigger_logger
        
        # Trigger for agent output → CLI output
        output_trigger_config = TriggerConfig(
            name="agent_output_trigger",
            trigger_type="data_updated"
        )
        self.components['output_trigger'] = DataUpdatedTrigger(
            [self.components['agent_output_du']], 
            output_trigger_config
        )
        
        output_trigger_logger = self.log_manager.get_logger("agent_output_trigger", "triggers")
        self.components['output_trigger'].nb_logger = output_trigger_logger
        
        # 6. Create direct links
        if self.log_manager.should_log_to_console:
            print("   Creating direct links...")
        self.main_logger.info("Creating DirectLinks")
        
        # Link: User Input DataUnit → Agent Input DataUnit
        user_to_agent_config = LinkConfig(
            link_type="direct"
        )
        self.components['user_to_agent_link'] = DirectLink(
            self.components['user_input_du'],
            self.components['agent_input_du'],
            user_to_agent_config,
            name="user_to_agent_link"
        )
        
        user_to_agent_logger = self.log_manager.get_logger("user_to_agent_link", "links")
        self.components['user_to_agent_link'].nb_logger = user_to_agent_logger
        
        # Link: Agent Input DataUnit → Agent Step
        agent_input_to_step_config = LinkConfig(
            link_type="direct"
        )
        self.components['agent_input_to_step_link'] = DirectLink(
            self.components['agent_input_du'],
            self.components['agent_step'],
            agent_input_to_step_config,
            name="agent_input_to_step_link"
        )
        
        agent_input_to_step_logger = self.log_manager.get_logger("agent_input_to_step_link", "links")
        self.components['agent_input_to_step_link'].nb_logger = agent_input_to_step_logger
        
        # Link: Agent Step → Agent Output DataUnit  
        step_to_output_config = LinkConfig(
            link_type="direct"
        )
        self.components['step_to_output_link'] = DirectLink(
            self.components['agent_step'],
            self.components['agent_output_du'],
            step_to_output_config,
            name="step_to_output_link"
        )
        
        step_to_output_logger = self.log_manager.get_logger("step_to_output_link", "links")
        self.components['step_to_output_link'].nb_logger = step_to_output_logger
        
        # 7. Set up trigger callbacks
        if self.log_manager.should_log_to_console:
            print("   Setting up trigger callbacks...")
        self.main_logger.info("Setting up trigger callbacks and starting links")
        
        # Start all links first
        await self.components['user_to_agent_link'].start()
        await self.components['agent_input_to_step_link'].start()
        await self.components['step_to_output_link'].start()
        
        # User input trigger → User to Agent link
        await self.components['user_trigger'].add_callback(
            self.components['user_to_agent_link'].transfer
        )
        
        # Agent input trigger → Agent step
        async def execute_agent_step(data):
            """Wrapper to execute agent step without passing data as positional argument."""
            await self.components['agent_step'].execute()
        
        await self.components['agent_trigger'].add_callback(execute_agent_step)
        
        # Start trigger monitoring
        if self.log_manager.should_log_to_console:
            print("   Starting trigger monitoring...")
        self.main_logger.info("Starting trigger monitoring")
        await self.components['user_trigger'].start_monitoring()
        await self.components['agent_trigger'].start_monitoring()
        await self.components['output_trigger'].start_monitoring()
        
        # 8. Create CLI interface
        if self.log_manager.should_log_to_console:
            print("   Creating CLI interface...")
        self.main_logger.info("Creating CLI interface")
        self.cli = LoadBalancedCLIInterface(
            self.components['user_input_du'],
            self.components['agent_output_du'],
            self.log_manager
        )
        
        log_workflow_event("parsl_workflow", "setup_complete", {
            "num_agents": len(self.agents),
            "executor_type": type(self.parsl_executor).__name__,
            "api_available": api_available,
            "components_created": len(self.components)
        })
        
        self.main_logger.info("Parsl Chat Workflow setup complete")
        if self.log_manager.should_log_to_console:
            print("✅ Parsl Chat workflow setup complete!")
    
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
            
            self.main_logger.info(f"Processing message with agent {agent_index}")
            
            # Process with selected agent - ConversationalAgent.process() expects a string
            start_time = time.time()
            result = await selected_agent.process(message)
            processing_time = time.time() - start_time
            
            # ConversationalAgent.process() returns a string directly
            response_text = result if isinstance(result, str) else str(result)
            
            self.main_logger.info(f"Message processed in {processing_time:.2f}s")
            return response_text
            
        except Exception as e:
            self.main_logger.error(f"Error processing message: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def run(self):
        """Run the Parsl chat workflow."""
        if not self.cli:
            raise RuntimeError("Workflow not set up. Call setup() first.")
        
        self.main_logger.info("Starting Parsl chat workflow execution")
        
        try:
            await self.cli.start()
            
            # Keep the workflow running until CLI stops
            while self.cli.running:
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self.main_logger.info("Workflow interrupted by user")
            if self.log_manager.should_log_to_console:
                print("\n🛑 Interrupted by user")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the workflow and cleanup resources."""
        log_workflow_event("parsl_workflow", "shutdown_start")
        self.main_logger.info("Starting Parsl chat workflow shutdown")
        if self.log_manager.should_log_to_console:
            print("\n🧹 Shutting down Parsl chat workflow...")
        
        if self.cli:
            await self.cli.stop()
        
        # Shutdown components in reverse order
        for name, component in reversed(list(self.components.items())):
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                    self.main_logger.debug(f"Shutdown component: {name}")
                    if self.log_manager.should_log_to_console:
                        print(f"   ✅ Shutdown {name}")
            except Exception as e:
                self.main_logger.error(f"Error shutting down {name}: {e}", 
                                     error_type=type(e).__name__)
                if self.log_manager.should_log_to_console:
                    print(f"   ❌ Error shutting down {name}: {e}")
        
        # Shutdown agents
        for i, agent in enumerate(self.agents):
            try:
                log_component_lifecycle("agents", f"parsl_agent_{i}", "shutdown")
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
                log_component_lifecycle("agents", f"parsl_agent_{i}", "shutdown_complete")
            except Exception as e:
                log_component_lifecycle("agents", f"parsl_agent_{i}", "shutdown_error", {"error": str(e)})
        
        # Shutdown Parsl executor
        if self.parsl_executor:
            try:
                log_component_lifecycle("executors", "parsl_executor", "shutdown")
                await self.parsl_executor.shutdown()
                log_component_lifecycle("executors", "parsl_executor", "shutdown_complete")
                self.main_logger.debug("Shutdown Parsl executor")
                if self.log_manager.should_log_to_console:
                    print("   ✅ Shutdown Parsl executor")
            except Exception as e:
                log_component_lifecycle("executors", "parsl_executor", "shutdown_error", {"error": str(e)})
                if self.log_manager.should_log_to_console:
                    print(f"   ❌ Error shutting down Parsl executor: {e}")
        
        # Create session summary
        try:
            summary = self.log_manager.create_session_summary()
            self.main_logger.info("Created session summary", 
                                 session_id=summary['session_id'],
                                 log_files_count=len(summary['log_files']))
            if self.log_manager.should_log_to_console:
                print(f"📊 Session summary created: {summary['session_id']}")
                if self.log_manager.should_log_to_file:
                    print(f"📁 Logs saved to: {self.log_manager.session_dir}")
            
            # Show performance summary
            performance = self.log_manager.get_performance_summary()
            if performance and self.log_manager.should_log_to_console:
                print("⚡ Performance metrics collected and saved to logs")
                
        except Exception as e:
            if self.log_manager.should_log_to_console:
                print(f"⚠️  Could not create session summary: {e}")
        
        # Clean up old logs
        try:
            self.log_manager.cleanup_old_logs(keep_days=7)
        except Exception as e:
            if self.log_manager.should_log_to_console:
                print(f"⚠️  Could not clean up old logs: {e}")
        
        log_workflow_event("parsl_workflow", "shutdown_complete")
        self.main_logger.info("Parsl chat workflow shutdown complete")
        if self.log_manager.should_log_to_console:
            print("✅ Shutdown complete!")


async def main():
    """Main entry point for the Parsl chat workflow demo."""
    
    # Enable debug mode for comprehensive logging
    set_debug_mode(True)
    
    # Create workflow first to get logging configuration
    workflow = ParslChatWorkflow()
    
    if workflow.log_manager.should_log_to_console:
        print("🚀 Starting NanoBrain Parsl Chat Workflow Demo")
        print("=" * 60)
    
    try:
        await workflow.setup()
        await workflow.run()
    except Exception as e:
        if hasattr(workflow, 'main_logger'):
            workflow.main_logger.error(f"Workflow error: {e}", error_type=type(e).__name__)
        if workflow.log_manager.should_log_to_console:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    finally:
        if hasattr(workflow, 'shutdown'):
            await workflow.shutdown()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 