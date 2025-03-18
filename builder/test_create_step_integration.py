import unittest
import asyncio
import os
import shutil
from unittest.mock import MagicMock, AsyncMock, patch

from builder.WorkflowSteps import CreateStep
from builder.NanoBrainBuilder import NanoBrainBuilder
from src.ExecutorBase import ExecutorBase
from src.DataStorageCommandLine import DataStorageCommandLine
from src.TriggerDataUpdated import TriggerDataUpdated


# Helper to create a properly managed AsyncMock
def create_safe_async_mock():
    """Create an AsyncMock that won't cause 'coroutine was never awaited' warnings."""
    mock = AsyncMock()
    # Set side_effect to a simple async function that returns None immediately
    async def safe_side_effect(*args, **kwargs):
        return None
    mock.side_effect = safe_side_effect
    return mock


def setup_llm_mocks():
    """Set up mocks for LLM initialization to avoid API key requirements."""
    # Set testing mode environment variable
    os.environ['NANOBRAIN_TESTING'] = '1'
    
    # Create mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke = AsyncMock(return_value="Mock LLM response")
    
    # Patch Agent._initialize_llm to return our mock
    llm_patcher = patch('src.Agent.Agent._initialize_llm', return_value=mock_llm)
    llm_mock = llm_patcher.start()
    
    # Patch Agent._load_prompt_template
    prompt_patcher = patch('src.Agent.Agent._load_prompt_template')
    prompt_mock = prompt_patcher.start()
    
    return llm_patcher, prompt_patcher, mock_llm


# Create a non-interactive version of start_monitoring
async def non_interactive_start_monitoring(self):
    """A non-interactive version of start_monitoring that automatically issues a 'finish' command."""
    if self._monitoring:
        return None  # Already monitoring
    
    self._monitoring = True
    
    # Show welcome message
    if hasattr(self, "welcome_message"):
        print(self.welcome_message)
    
    # Add important attributes if missing
    if not hasattr(self, "_add_to_history"):
        self._add_to_history = MagicMock()
    
    if not hasattr(self, "_force_output_change"):
        self._force_output_change = MagicMock()
    
    if not hasattr(self, "process"):
        self.process = AsyncMock(return_value=True)
    
    # Set up output if missing
    if not hasattr(self, "output"):
        from unittest.mock import MagicMock
        output_mock = MagicMock()
        output_mock.get = MagicMock(return_value="test output")
        output_mock.set = MagicMock()
        self.output = output_mock
    
    # Automatically send 'finish' command
    self._monitoring = False
    response = "Step creation completed."
    
    # Call methods safely
    self._add_to_history("finish", response)
    self._force_output_change(response)
    
    print("\n✅ Step creation session ended (non-interactive mode). Files will be saved.")
    return None


class TestCreateStepIntegration(unittest.TestCase):
    """Integration test for CreateStep"""

    def setUp(self):
        """Set up the test environment"""
        # Set up LLM mocks
        self.llm_patcher, self.prompt_patcher, self.mock_llm = setup_llm_mocks()
        
        # Create a test workflow directory
        self.test_workflow_dir = "/tmp/test_workflow"
        os.makedirs(self.test_workflow_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_workflow_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.test_workflow_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(self.test_workflow_dir, "test"), exist_ok=True)
        
        # Create a test executor
        self.executor = MagicMock(spec=ExecutorBase)
        
        # Create a properly mocked builder rather than a real NanoBrainBuilder
        self.builder = MagicMock()
        self.builder.executor = self.executor
        self.builder.get_current_workflow = MagicMock(return_value=self.test_workflow_dir)
        self.builder._debug_mode = False
        self.builder.debug_mode = False
        
        # Create agent with tools
        self.builder.agent = MagicMock()
        
        # Create file_writer_tool
        self.file_writer_tool = MagicMock()
        self.file_writer_tool.__class__ = MagicMock()
        self.file_writer_tool.__class__.__name__ = 'StepFileWriter'
        
        # Add proper create_file method
        async def mock_create_file(path, content):
            # Create parent directories
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Write file content
            with open(path, 'w') as f:
                f.write(content)
            return True
            
        self.file_writer_tool.create_file = mock_create_file
        
        # Add tools to agent
        self.builder.agent.tools = [self.file_writer_tool]
        
        # Add config
        self.builder.config = {
            'defaults': {
                'workflows_dir': '/tmp'
            }
        }

    def tearDown(self):
        """Clean up after the test"""
        # Stop the patchers
        self.llm_patcher.stop()
        self.prompt_patcher.stop()
        
        # Remove the test directory
        if os.path.exists(self.test_workflow_dir):
            shutil.rmtree(self.test_workflow_dir)
    
    def test_create_step_execute(self):
        """Test the CreateStep.execute method"""
        
        async def run_test():
            # Clean up the step directory if it exists
            step_dir = os.path.join(self.test_workflow_dir, "src", "StepTestStep")
            if os.path.exists(step_dir):
                shutil.rmtree(step_dir)
            
            # Create AsyncMock for generate_step_template with a defined return value
            template_mock = AsyncMock(return_value="// Generated template code")
            
            # Create AsyncMock for create_file
            create_file_mock = AsyncMock(return_value=True)
            
            # Create a proper mock for DataStorageCommandLine
            command_line_mock = MagicMock()
            command_line_mock.start_monitoring = AsyncMock(side_effect=non_interactive_start_monitoring)
            command_line_mock._monitoring = False
            command_line_mock.welcome_message = "Welcome to command line mock"
            command_line_mock.prompt = "test> "
            command_line_mock.exit_command = "exit"
            command_line_mock.name = "CommandLineMock"  # Add name to avoid attribute errors
            # Add methods that might be called by the code
            command_line_mock._add_to_history = MagicMock()
            command_line_mock._force_output_change = MagicMock()
            command_line_mock.process = AsyncMock(return_value=True)
            
            # Create a mock for DataUnitString for output
            output_mock = MagicMock()
            output_mock.get = MagicMock(return_value="test output")
            output_mock.set = MagicMock()
            command_line_mock.output = output_mock
            
            # Create a mock for LinkDirect
            link_mock = MagicMock()
            link_mock.start_monitoring = AsyncMock(return_value=None)
            link_mock._monitoring = False
            link_mock._debug_mode = False
            link_mock.debug_mode = False
            link_mock.name = "LinkMock"  # Add name to avoid attribute errors
            
            # Create a properly mocked TriggerDataUpdated
            trigger_mock = MagicMock()
            trigger_mock.start_monitoring = MagicMock(return_value=None)  # Not async method
            trigger_mock._monitoring = False
            trigger_mock._debug_mode = False
            trigger_mock.debug_mode = False
            # Add necessary attributes for TriggerDataUpdated
            trigger_mock.source_step = command_line_mock
            trigger_mock.runnable = link_mock
            
            # Create a mock for the file writer tool
            file_writer_mock = MagicMock()
            file_writer_mock.create_file = create_file_mock
            file_writer_mock.__class__ = MagicMock()
            file_writer_mock.__class__.__name__ = 'StepFileWriter'
            
            # Create builder mock
            builder_mock = MagicMock()
            builder_mock.generate_step_template = template_mock
            builder_mock.get_generated_code = MagicMock(return_value="test code")
            builder_mock.get_generated_config = MagicMock(return_value="test config")
            builder_mock.get_generated_tests = MagicMock(return_value="test tests")
            builder_mock._debug_mode = False
            builder_mock.name = "BuilderMock"  # Add name to avoid attribute errors
            builder_mock.input_storage = None
            builder_mock.input_sources = {}
            builder_mock._monitoring = False
            
            # Add file_writer_tool to our builder
            self.builder.agent = MagicMock()
            self.builder.agent.tools = [file_writer_mock]
                
            # Set up all the patches needed to make the test non-interactive
            with patch('src.DataStorageCommandLine.DataStorageCommandLine', return_value=command_line_mock), \
                 patch('builder.AgentWorkflowBuilder.AgentWorkflowBuilder', return_value=builder_mock), \
                 patch('src.LinkDirect.LinkDirect', return_value=link_mock), \
                 patch('src.TriggerDataUpdated.TriggerDataUpdated', return_value=trigger_mock), \
                 patch('builder.WorkflowSteps.ConfigManager') as mock_config_manager, \
                 patch('asyncio.create_task') as mock_create_task, \
                 patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                
                # Make sleep a no-op to speed up the test
                async def no_sleep(*args, **kwargs):
                    return None
                mock_sleep.side_effect = no_sleep
                
                # Configure ConfigManager to return our mocks
                mock_config_instance = MagicMock()
                def create_instance_side_effect(*args, **kwargs):
                    if args[0] == "AgentWorkflowBuilder":
                        return builder_mock
                    elif args[0] == "DataStorageCommandLine":
                        return command_line_mock
                    elif args[0] == "LinkDirect":
                        return link_mock
                    elif args[0] == "TriggerDataUpdated":
                        return trigger_mock
                    else:
                        return MagicMock()
                
                mock_config_instance.create_instance = MagicMock(side_effect=create_instance_side_effect)
                mock_config_manager.return_value = mock_config_instance
                
                # Configure create_task to just run the coroutine to avoid multiple event loops
                async def run_coroutine(coro):
                    try:
                        if asyncio.iscoroutine(coro):
                            return await coro
                        return coro
                    except Exception as e:
                        print(f"Error in coroutine: {e}")
                        return None
                
                mock_create_task.side_effect = lambda coro: asyncio.ensure_future(run_coroutine(coro))
                
                # Patch the run_in_executor for input to avoid stdin reads
                original_run_in_executor = asyncio.get_event_loop().run_in_executor
                
                async def mock_run_in_executor(executor, func, *args):
                    if func == input:
                        # Return 'finish' for any input request
                        return 'finish'
                    return await original_run_in_executor(executor, func, *args)
                
                # Apply our patch to asyncio.get_event_loop().run_in_executor
                with patch('asyncio.get_event_loop') as mock_get_event_loop:
                    mock_loop = MagicMock()
                    mock_loop.run_in_executor = mock_run_in_executor
                    mock_get_event_loop.return_value = mock_loop
                    
                    # Execute the CreateStep
                    print("Before CreateStep.execute call")
                    result = await CreateStep.execute(
                        self.builder, 
                        "test_step",
                        command_line=command_line_mock,
                        agent_builder=builder_mock
                    )
                    
                    print(f"Result: {result}")
                    
                    # Verify the result
                    self.assertTrue(result.get("success", False),
                                   f"CreateStep execution failed: {result.get('error', 'Unknown error')}")
                    
                    # Check that the step directory was created
                    self.assertTrue(os.path.exists(step_dir),
                                  f"Step directory was not created: {step_dir}")
                    
                    # Verify create_file was called
                    self.assertTrue(create_file_mock.called, "The create_file mock was not called")
                    
                    return result
        
        # Run the test using a dedicated event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Set a timeout to prevent the test from hanging
            result = loop.run_until_complete(asyncio.wait_for(run_test(), timeout=10.0))  # Increased timeout
            self.assertTrue(result.get("success", False))
            
            # Ensure all pending tasks are complete to avoid "Task was destroyed but it is pending" warnings
            pending = asyncio.all_tasks(loop)
            for task in pending:
                if not task.done():
                    # Cancel all pending tasks immediately
                    task.cancel()
                    try:
                        # Wait for cancellation to complete with a short timeout
                        loop.run_until_complete(asyncio.wait_for(task, timeout=0.5))  # Increased timeout
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
        except asyncio.TimeoutError:
            # If test times out, print a helpful message and fail
            self.fail("Test timed out, likely stuck in an interactive prompt")
        finally:
            # Close the event loop
            loop.close()


if __name__ == "__main__":
    unittest.main() 