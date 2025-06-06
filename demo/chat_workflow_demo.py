#!/usr/bin/env python3
"""
NanoBrain Chat Workflow Demo

A comprehensive demonstration of the NanoBrain framework featuring:
- CLI input/output interface
- Conversational agent with LLM processing
- Data units for state management
- Triggers for event-driven processing
- Direct links for data flow
- Step wrapper for workflow integration
- Comprehensive file logging with debug information

Architecture:
CLI Input → User Input DataUnit → Agent Input DataUnit → ConversationalAgentStep → Agent Output DataUnit → CLI Output

Each arrow represents a DirectLink, and DataUpdatedTriggers activate processing at each stage.
"""

import asyncio
import sys
import os
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import NanoBrain components
from core.data_unit import DataUnitMemory, DataUnitConfig
from core.trigger import DataUpdatedTrigger, TriggerConfig
from core.link import DirectLink, LinkConfig
from core.step import Step, StepConfig
from core.agent import ConversationalAgent, AgentConfig
from core.executor import LocalExecutor, ExecutorConfig
from config.component_factory import ComponentFactory, get_factory

# Import logging system
from core.logging_system import NanoBrainLogger, get_logger, set_debug_mode, OperationType

# Import global configuration
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
    from config_manager import get_config_manager, get_api_key, get_provider_config, get_logging_config, should_log_to_file, should_log_to_console
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import configuration manager: {e}")
    print("   API keys will need to be set via environment variables")
    CONFIG_AVAILABLE = False


class LogManager:
    """
    Manages comprehensive file logging for the chat workflow demo.
    
    Creates organized log files with rotation and provides utilities for log analysis.
    Respects global logging configuration including console/file mode settings.
    """
    
    def __init__(self, base_log_dir: Optional[str] = None):
        # Get global logging configuration
        try:
            self.logging_config = get_logging_config()
            self.should_log_to_file = should_log_to_file()
            self.should_log_to_console = should_log_to_console()
            
            # Get file logging configuration
            file_config = self.logging_config.get('file', {})
            if base_log_dir is None:
                base_log_dir = file_config.get('base_directory', 'logs')
            self.use_session_directories = file_config.get('use_session_directories', True)
            
        except ImportError:
            # Fallback if config manager not available
            self.logging_config = {}
            self.should_log_to_file = True
            self.should_log_to_console = True
            if base_log_dir is None:
                base_log_dir = "logs"
            self.use_session_directories = True
        
        self.base_log_dir = Path(base_log_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_session_directories:
            self.session_dir = self.base_log_dir / f"session_{self.session_id}"
        else:
            self.session_dir = self.base_log_dir
            
        self.loggers = {}
        
        # Create log directory structure only if file logging is enabled
        if self.should_log_to_file:
            self._setup_log_directories()
        
    def _setup_log_directories(self):
        """Create organized log directory structure."""
        directories = [
            self.session_dir,
            self.session_dir / "components",
            self.session_dir / "agents", 
            self.session_dir / "steps",
            self.session_dir / "data_units",
            self.session_dir / "triggers",
            self.session_dir / "links"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Only print directory creation message if console logging is enabled
        if self.should_log_to_console:
            print(f"📁 Created log session directory: {self.session_dir}")
        
    def get_logger(self, name: str, category: str = "components", 
                   debug_mode: bool = True) -> NanoBrainLogger:
        """
        Get or create a logger for a specific component.
        
        Args:
            name: Logger name
            category: Log category (components, agents, steps, etc.)
            debug_mode: Enable debug logging
            
        Returns:
            Configured NanoBrainLogger instance
        """
        logger_key = f"{category}_{name}"
        
        if logger_key not in self.loggers:
            # Determine log file path
            log_file = None
            if self.should_log_to_file:
                log_file = self.session_dir / category / f"{name}.log"
            
            # Create logger with global configuration
            logger = NanoBrainLogger(
                name=f"{category}.{name}",
                log_file=log_file,
                debug_mode=debug_mode
                # enable_console and enable_file will be determined automatically from global config
            )
            
            self.loggers[logger_key] = logger
            
            # Log the logger creation only if appropriate
            if self.should_log_to_console or self.should_log_to_file:
                logger.info(f"Logger initialized for {name}", 
                           category=category, 
                           log_file=str(log_file) if log_file else "none",
                           session_id=self.session_id,
                           console_enabled=self.should_log_to_console,
                           file_enabled=self.should_log_to_file)
        
        return self.loggers[logger_key]
    
    def get_main_logger(self) -> NanoBrainLogger:
        """Get the main workflow logger."""
        return self.get_logger("chat_workflow", "components")
    
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from all loggers."""
        performance_data = {}
        
        for logger_key, logger in self.loggers.items():
            if hasattr(logger, 'get_performance_summary'):
                performance_data[logger_key] = logger.get_performance_summary()
                
        return performance_data
    
    def cleanup_old_logs(self, keep_days: int = 7):
        """Clean up old log sessions."""
        if not self.base_log_dir.exists():
            return
            
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        
        for session_dir in self.base_log_dir.glob("session_*"):
            if session_dir.is_dir():
                try:
                    if session_dir.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(session_dir)
                        print(f"🗑️  Cleaned up old log session: {session_dir.name}")
                except Exception as e:
                    print(f"⚠️  Could not clean up {session_dir}: {e}")


class ConversationalAgentStep(Step):
    """
    Step wrapper for ConversationalAgent to integrate with NanoBrain workflow.
    
    This step processes user input through a conversational agent and returns responses.
    """
    
    def __init__(self, config: StepConfig, agent: ConversationalAgent, log_manager: LogManager):
        super().__init__(config)
        self.agent = agent
        self.conversation_count = 0
        self.log_manager = log_manager
        
        # Create dedicated logger for this step
        self.step_logger = log_manager.get_logger("conversational_agent_step", "steps")
        
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input through the conversational agent.
        
        Args:
            inputs: Dictionary containing 'user_input' key with user message
            
        Returns:
            Dictionary containing 'agent_response' key with agent's response
        """
        # Extract user input - handle nested dictionary structure
        user_input_data = inputs.get('user_input', '')
        
        # If user_input is a dictionary, extract the actual message
        if isinstance(user_input_data, dict):
            user_input = user_input_data.get('user_input', '')
        else:
            user_input = user_input_data
        
        # Ensure user_input is a string
        if not isinstance(user_input, str):
            user_input = str(user_input) if user_input else ''
            
        if not user_input or user_input.strip() == '':
            self.step_logger.warning("Empty user input received")
            return {'agent_response': ''}
        
        self.conversation_count += 1
        
        # Log the processing start
        self.step_logger.info(f"Processing conversation #{self.conversation_count}", 
                             user_input_preview=user_input[:100] + "..." if len(user_input) > 100 else user_input,
                             input_length=len(user_input))
        
        try:
            # Process through conversational agent
            start_time = time.time()
            response = await self.agent.process(user_input)
            processing_time = (time.time() - start_time) * 1000
            
            self.step_logger.info(f"Completed conversation #{self.conversation_count}", 
                                 user_input_length=len(user_input),
                                 response_length=len(response) if response else 0,
                                 processing_time_ms=processing_time)
            
            # Log conversation details
            self.step_logger.debug("Conversation details",
                                  conversation_id=self.conversation_count,
                                  user_input=user_input,
                                  agent_response=response,
                                  processing_time_ms=processing_time)
            
            return {'agent_response': response or 'I apologize, but I could not generate a response.'}
            
        except Exception as e:
            self.step_logger.error(f"Error processing user input: {e}", 
                                  error_type=type(e).__name__,
                                  conversation_id=self.conversation_count,
                                  user_input_length=len(user_input))
            return {'agent_response': f'Sorry, I encountered an error: {str(e)}'}


class CLIInterface:
    """
    Command Line Interface for the chat workflow.
    
    Handles user input and output, interfacing with NanoBrain data units.
    """
    
    def __init__(self, input_data_unit: DataUnitMemory, output_data_unit: DataUnitMemory, log_manager: LogManager):
        self.input_data_unit = input_data_unit
        self.output_data_unit = output_data_unit
        self.log_manager = log_manager
        self.running = False
        self.input_thread = None
        self.output_monitor_task = None
        self.input_queue = asyncio.Queue()
        self.last_output_time = 0.0
        self.event_loop = None  # Store reference to the event loop
        
        # Create dedicated logger for CLI
        self.cli_logger = log_manager.get_logger("cli_interface", "components")
        
    async def start(self):
        """Start the CLI interface."""
        self.running = True
        
        # Store the current event loop for use in the input thread
        self.event_loop = asyncio.get_running_loop()
        
        # Start output monitoring task
        self.output_monitor_task = asyncio.create_task(self._monitor_output())
        
        # Start input processing task
        self.input_processor_task = asyncio.create_task(self._process_input_queue())
        
        self.cli_logger.info("CLI interface started")
        
        # Only show welcome messages if console logging is enabled
        if self.log_manager.should_log_to_console:
            print("🧠 NanoBrain Chat Workflow Demo")
            print("=" * 50)
            print("Type your messages below. Type 'quit', 'exit', or 'bye' to stop.")
            print("Type 'help' for available commands.")
            print("Type 'logs' to show log information.")
            print("=" * 50)
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
    async def stop(self):
        """Stop the CLI interface."""
        self.running = False
        
        self.cli_logger.info("Stopping CLI interface")
        
        # Cancel output monitoring
        if self.output_monitor_task and not self.output_monitor_task.done():
            self.output_monitor_task.cancel()
            try:
                await self.output_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel input processing
        if hasattr(self, 'input_processor_task') and not self.input_processor_task.done():
            self.input_processor_task.cancel()
            try:
                await self.input_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
            
        self.cli_logger.info("CLI interface stopped")
    
    async def _monitor_output(self):
        """Monitor output data unit for new responses."""
        try:
            while self.running:
                # Check if output data unit has been updated
                if hasattr(self.output_data_unit, 'get_metadata'):
                    current_update_time = await self.output_data_unit.get_metadata('last_updated', 0.0)
                    
                    if current_update_time > self.last_output_time:
                        self.last_output_time = current_update_time
                        data = await self.output_data_unit.get()
                        if data:
                            await self._on_output_received(data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.cli_logger.error(f"Error monitoring output: {e}", error_type=type(e).__name__)
            print(f"❌ Error monitoring output: {e}")
    
    async def _process_input_queue(self):
        """Process input from the queue."""
        try:
            while self.running:
                try:
                    # Wait for input with timeout
                    user_input = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                    
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("👋 Goodbye!")
                        self.cli_logger.info("User requested exit")
                        self.running = False
                        break
                        
                    if user_input.lower() == 'help':
                        self._show_help()
                        continue
                        
                    if user_input.lower() == 'logs':
                        self._show_log_info()
                        continue
                    
                    # Log user input
                    self.cli_logger.info("User input received", 
                                        input_length=len(user_input),
                                        input_preview=user_input[:50] + "..." if len(user_input) > 50 else user_input)
                    
                    # Send input to data unit (this will trigger the workflow)
                    await self.input_data_unit.set({'user_input': user_input})
                    
                except asyncio.TimeoutError:
                    # No input received, continue loop
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.cli_logger.error(f"Error processing input: {e}", error_type=type(e).__name__)
            print(f"❌ Error processing input: {e}")
        
    def _input_loop(self):
        """Input loop running in separate thread."""
        while self.running:
            try:
                # Handle input differently based on console logging mode
                if self.log_manager.should_log_to_console:
                    user_input = input("\n👤 You: ").strip()
                else:
                    # In file-only mode, provide a simple prompt
                    user_input = input("Input: ").strip()
                
                if not user_input:
                    continue
                
                # Put input into queue for async processing
                if self.event_loop and not self.event_loop.is_closed():
                    try:
                        # Use the stored event loop reference
                        future = asyncio.run_coroutine_threadsafe(
                            self.input_queue.put(user_input), 
                            self.event_loop
                        )
                        # Wait for the coroutine to complete with a timeout
                        future.result(timeout=1.0)
                    except Exception as e:
                        self.cli_logger.error(f"Error queuing input: {e}", error_type=type(e).__name__)
                        if self.log_manager.should_log_to_console:
                            print(f"❌ Error queuing input: {e}")
                        self.running = False
                        break
                else:
                    if self.log_manager.should_log_to_console:
                        print("❌ Event loop not available")
                    self.cli_logger.error("Event loop not available for input processing")
                    self.running = False
                    break
                
            except (EOFError, KeyboardInterrupt):
                if self.log_manager.should_log_to_console:
                    print("\n👋 Goodbye!")
                self.cli_logger.info("CLI interrupted by user")
                self.running = False
                break
            except Exception as e:
                self.cli_logger.error(f"Input error: {e}", error_type=type(e).__name__)
                if self.log_manager.should_log_to_console:
                    print(f"❌ Input error: {e}")
    
    async def _on_output_received(self, data: Dict[str, Any]):
        """Handle output from the agent."""
        response = data.get('agent_response', '')
        if response and response.strip():
            self.cli_logger.info("Agent response received", 
                                response_length=len(response),
                                response_preview=response[:100] + "..." if len(response) > 100 else response)
            # Only print to console if console logging is enabled
            if self.log_manager.should_log_to_console:
                print(f"\n🤖 Assistant: {response}")
    
    def _show_help(self):
        """Show help information."""
        # Only show help if console logging is enabled
        if not self.log_manager.should_log_to_console:
            return
            
        print("\n📋 Available Commands:")
        print("  help     - Show this help message")
        print("  logs     - Show logging information")
        print("  quit     - Exit the chat")
        print("  exit     - Exit the chat") 
        print("  bye      - Exit the chat")
        print("\n💡 Tips:")
        print("  - Ask questions about any topic")
        print("  - The assistant maintains conversation context")
        print("  - All interactions are processed through the NanoBrain framework")
        print("  - Debug information is logged to files for analysis")
        
    def _show_log_info(self):
        """Show logging information."""
        # Only show log info if console logging is enabled
        if not self.log_manager.should_log_to_console:
            return
            
        print(f"\n📊 Logging Information:")
        print(f"  Logging Mode: {self.log_manager.logging_config.get('mode', 'both')}")
        print(f"  Console Logging: {'Enabled' if self.log_manager.should_log_to_console else 'Disabled'}")
        print(f"  File Logging: {'Enabled' if self.log_manager.should_log_to_file else 'Disabled'}")
        print(f"  Session ID: {self.log_manager.session_id}")
        
        if self.log_manager.should_log_to_file:
            print(f"  Log Directory: {self.log_manager.session_dir}")
            print(f"  Active Loggers: {len(self.log_manager.loggers)}")
            
            # Show log files
            log_files = list(self.log_manager.session_dir.rglob("*.log"))
            if log_files:
                print(f"  Log Files ({len(log_files)}):")
                for log_file in log_files[:5]:  # Show first 5
                    size = log_file.stat().st_size if log_file.exists() else 0
                    print(f"    - {log_file.name} ({size} bytes)")
                if len(log_files) > 5:
                    print(f"    ... and {len(log_files) - 5} more files")
            else:
                print("  No log files created yet")
        else:
            print("  File logging is disabled - no log files created")


class ChatWorkflow:
    """
    Main chat workflow orchestrator.
    
    Sets up and manages the complete NanoBrain workflow for chat functionality.
    """
    
    def __init__(self):
        self.factory = get_factory()
        self.components = {}
        self.cli = None
        self.executor = None
        self.config_manager = None
        
        # Initialize log manager
        self.log_manager = LogManager()
        self.main_logger = self.log_manager.get_main_logger()
        
        # Enable debug mode globally
        set_debug_mode(True)
        
    def _setup_api_keys(self):
        """Setup API keys from global configuration."""
        self.main_logger.info("Setting up API keys from global configuration")
        
        if not CONFIG_AVAILABLE:
            self.main_logger.warning("Configuration manager not available, using environment variables")
            if self.log_manager.should_log_to_console:
                print("   ⚠️  Configuration manager not available, using environment variables")
            return
            
        try:
            self.config_manager = get_config_manager()
            
            # Get OpenAI API key and set environment variable if not already set
            openai_key = get_api_key('openai')
            if openai_key and not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = openai_key
                self.main_logger.info("OpenAI API key loaded from configuration")
                if self.log_manager.should_log_to_console:
                    print("   ✅ OpenAI API key loaded from configuration")
            elif openai_key:
                self.main_logger.info("OpenAI API key already set in environment")
                if self.log_manager.should_log_to_console:
                    print("   ✅ OpenAI API key already set in environment")
            else:
                self.main_logger.warning("No OpenAI API key found in configuration")
                if self.log_manager.should_log_to_console:
                    print("   ⚠️  No OpenAI API key found in configuration")
                
            # Get Anthropic API key and set environment variable if not already set
            anthropic_key = get_api_key('anthropic')
            if anthropic_key and not os.getenv('ANTHROPIC_API_KEY'):
                os.environ['ANTHROPIC_API_KEY'] = anthropic_key
                self.main_logger.info("Anthropic API key loaded from configuration")
                if self.log_manager.should_log_to_console:
                    print("   ✅ Anthropic API key loaded from configuration")
            elif anthropic_key:
                self.main_logger.info("Anthropic API key already set in environment")
                if self.log_manager.should_log_to_console:
                    print("   ✅ Anthropic API key already set in environment")
            else:
                self.main_logger.warning("No Anthropic API key found in configuration")
                if self.log_manager.should_log_to_console:
                    print("   ⚠️  No Anthropic API key found in configuration")
                
            # Show available providers
            if self.config_manager:
                available_providers = self.config_manager.get_available_providers()
                if available_providers:
                    self.main_logger.info(f"Available AI providers: {', '.join(available_providers)}")
                    if self.log_manager.should_log_to_console:
                        print(f"   📋 Available AI providers: {', '.join(available_providers)}")
                else:
                    self.main_logger.warning("No AI providers with valid API keys found")
                    if self.log_manager.should_log_to_console:
                        print("   ⚠️  No AI providers with valid API keys found")
                    
        except Exception as e:
            self.main_logger.error(f"Error setting up API keys: {e}", error_type=type(e).__name__)
            if self.log_manager.should_log_to_console:
                print(f"   ❌ Error setting up API keys: {e}")
        
    async def setup(self):
        """Set up the complete chat workflow."""
        self.main_logger.info("Starting NanoBrain Chat Workflow setup")
        
        # Only show setup messages if console logging is enabled
        if self.log_manager.should_log_to_console:
            print("🔧 Setting up NanoBrain Chat Workflow...")
        
        # 0. Setup API keys from global configuration
        if self.log_manager.should_log_to_console:
            print("   Setting up API keys...")
        self._setup_api_keys()
        
        # 1. Create executor
        if self.log_manager.should_log_to_console:
            print("   Creating executor...")
        self.main_logger.info("Creating LocalExecutor")
        executor_config = ExecutorConfig(
            executor_type="local",
            max_workers=2,
            timeout=30.0
        )
        self.executor = LocalExecutor(executor_config)
        await self.executor.initialize()
        
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
        
        # 3. Create conversational agent
        if self.log_manager.should_log_to_console:
            print("   Creating conversational agent...")
        self.main_logger.info("Creating ConversationalAgent")
        
        # Determine which model to use based on available API keys
        model = "gpt-3.5-turbo"  # Default
        if CONFIG_AVAILABLE and self.config_manager:
            # Check if we have OpenAI key
            if get_api_key('openai'):
                model = "gpt-3.5-turbo"
                self.main_logger.info(f"Using OpenAI model: {model}")
                print(f"   Using OpenAI model: {model}")
            # Fallback to Anthropic if available
            elif get_api_key('anthropic'):
                model = "claude-3-haiku-20240307"
                self.main_logger.info(f"Using Anthropic model: {model}")
                print(f"   Using Anthropic model: {model}")
            else:
                self.main_logger.warning(f"No API keys available, using default model: {model}")
                print(f"   ⚠️  No API keys available, using default model: {model}")
        
        agent_config = AgentConfig(
            name="ChatAssistant",
            description="Helpful conversational assistant for chat workflow",
            model=model,
            temperature=0.7,
            max_tokens=1000,
            system_prompt="""You are a helpful and friendly AI assistant. You engage in natural conversations with users, providing helpful, accurate, and thoughtful responses. 

Key guidelines:
- Be conversational and personable
- Provide clear and helpful answers
- Ask follow-up questions when appropriate
- Maintain context throughout the conversation
- Be concise but thorough in your responses""",
            auto_initialize=False,  # We'll initialize manually
            debug_mode=True,
            enable_logging=True,
            log_conversations=True
        )
        
        agent = ConversationalAgent(agent_config, executor=self.executor)
        
        # Create agent logger
        agent_logger = self.log_manager.get_logger("chat_assistant", "agents")
        agent.nb_logger = agent_logger  # Override the default logger
        
        await agent.initialize()
        
        # 4. Create conversational agent step
        if self.log_manager.should_log_to_console:
            print("   Creating agent step...")
        self.main_logger.info("Creating ConversationalAgentStep")
        step_config = StepConfig(
            name="chat_agent_step",
            description="Conversational agent step for chat processing",
            debug_mode=True
        )
        
        self.components['agent_step'] = ConversationalAgentStep(
            step_config, 
            agent,
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
        
        # Create trigger logger
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
        
        # Agent input trigger → Agent step (with wrapper to handle data parameter)
        async def execute_agent_step(data):
            """Wrapper to execute agent step without passing data as positional argument."""
            await self.components['agent_step'].execute()
        
        await self.components['agent_trigger'].add_callback(execute_agent_step)
        
        # Agent output trigger → CLI (handled by CLI interface)
        # CLI will subscribe directly to the output data unit
        
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
        self.cli = CLIInterface(
            self.components['user_input_du'],
            self.components['agent_output_du'],
            self.log_manager
        )
        
        self.main_logger.info("Chat workflow setup complete")
        if self.log_manager.should_log_to_console:
            print("✅ Chat workflow setup complete!")
        
    async def run(self):
        """Run the chat workflow."""
        if not self.cli:
            raise RuntimeError("Workflow not set up. Call setup() first.")
        
        self.main_logger.info("Starting chat workflow execution")
        
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
        self.main_logger.info("Starting chat workflow shutdown")
        if self.log_manager.should_log_to_console:
            print("\n🧹 Shutting down chat workflow...")
        
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
        
        if self.executor:
            await self.executor.shutdown()
            self.main_logger.debug("Shutdown executor")
            if self.log_manager.should_log_to_console:
                print("   ✅ Shutdown executor")
        
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
        
        self.main_logger.info("Chat workflow shutdown complete")
        if self.log_manager.should_log_to_console:
            print("✅ Shutdown complete!")


async def main():
    """Main entry point for the chat workflow demo."""
    # Create workflow first to get logging configuration
    workflow = ChatWorkflow()
    
    if workflow.log_manager.should_log_to_console:
        print("🚀 Starting NanoBrain Chat Workflow Demo")
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
    # Run the chat workflow
    asyncio.run(main()) 