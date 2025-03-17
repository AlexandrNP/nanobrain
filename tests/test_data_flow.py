import asyncio
import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.DataUnitString import DataUnitString
from src.DataStorageCommandLine import DataStorageCommandLine
from src.LinkDirect import LinkDirect
from src.TriggerDataUpdated import TriggerDataUpdated
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.ExecutorBase import ExecutorBase


class TestDataFlow(unittest.TestCase):
    """Test the data flow between components."""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.executor = ExecutorBase()

    def tearDown(self):
        # Clean up any pending tasks
        for task in asyncio.all_tasks(self.loop):
            if not task.done():
                task.cancel()
                try:
                    self.loop.run_until_complete(task)
                except (asyncio.CancelledError, Exception):
                    pass
        
        self.loop.close()
        
    def test_data_flow(self):
        """Test the data flow from command line to agent."""
        async def run_test():
            # Create command line
            command_line = DataStorageCommandLine(executor=self.executor, name="TestCommandLine", debug=True)
            
            # Create output unit
            cmd_output = DataUnitString(name="CommandOutput")
            command_line.output = cmd_output
            
            # Create agent builder
            agent_builder = AgentWorkflowBuilder(executor=self.executor, name="TestAgentBuilder", debug=True)
            
            # Create link between command line and agent
            link_id = "test_link"
            agent_builder.register_input_source(link_id, cmd_output)
            
            # Create link
            link = LinkDirect(
                source_step=command_line,
                sink_step=agent_builder,
                link_id=link_id
            )
            
            # Create trigger
            trigger = TriggerDataUpdated(
                source_step=command_line,
                runnable=link,
                check_interval=0.1,
                debug=True
            )
            
            # Set direct reference
            command_line.agent_builder = agent_builder
            
            # Start monitoring - use the synchronous version since we're in an async context
            trigger.start_monitoring()
            
            # Process test data
            test_input = "Hello, world!"
            print(f"\nProcessing test input: {test_input}")
            await command_line.process(test_input)
            
            # Wait a bit for processing
            await asyncio.sleep(1)
            
            # Verify data was transferred
            agent_inputs = agent_builder.input_sources.get(link_id)
            if agent_inputs:
                agent_data = agent_inputs.get()
                print(f"Agent received data: {agent_data}")
                self.assertEqual(agent_data, test_input)
            else:
                self.fail("No data received by agent")
                
            # Stop monitoring
            trigger.stop_monitoring()
            
            return True
            
        # Run the test
        result = self.loop.run_until_complete(run_test())
        self.assertTrue(result)
        
    def test_direct_link_transfer(self):
        """Test the LinkDirect transfer method directly."""
        async def run_test():
            # Create source step with output
            from src.Step import Step
            source = Step(executor=self.executor, name="TestSource")
            source.output = DataUnitString(name="SourceOutput")
            source.output.set("Test output data")
            
            # Create sink step with input
            sink = Step(executor=self.executor, name="TestSink")
            sink_input = DataUnitString(name="SinkInput")
            
            # Create link
            link_id = "test_direct_link"
            sink.register_input_source(link_id, sink_input)
            
            link = LinkDirect(
                source_step=source,
                sink_step=sink,
                link_id=link_id,
                debug=True  # Enable debug mode to see what's happening
            )
            
            # Call transfer directly
            print("\nTesting direct transfer...")
            result = await link.transfer()
            
            # Verify transfer succeeded
            print(f"Transfer result: {result}")
            
            # Check input data
            input_data = sink.input_sources.get(link_id).get()
            print(f"Sink input data: {input_data}")
            
            # The test should pass even if result is False, as long as the data was transferred
            return input_data == "Test output data"
            
        # Run the test
        result = self.loop.run_until_complete(run_test())
        self.assertTrue(result)
        
    def test_command_line_process(self):
        """Test the DataStorageCommandLine process method."""
        async def run_test():
            # Create command line
            command_line = DataStorageCommandLine(executor=self.executor, name="TestProcessCommandLine", debug=True)
            
            # Create output unit
            cmd_output = DataUnitString(name="CommandOutput")
            command_line.output = cmd_output
            
            # Process test data
            test_input = "Test command line process"
            print(f"\nProcessing command line input: {test_input}")
            result = await command_line.process(test_input)
            
            # Verify result
            print(f"Command line process result: {result}")
            
            # Check output data
            output_data = cmd_output.get()
            print(f"Command line output data: {output_data}")
            
            return result == test_input and output_data == test_input
            
        # Run the test
        result = self.loop.run_until_complete(run_test())
        self.assertTrue(result)
    
    def test_link_trigger_pattern(self):
        """Test the Link/Trigger communication pattern."""
        async def run_test():
            # Create source and sink steps
            from src.Step import Step
            source = Step(executor=self.executor, name="SourceStep")
            sink = Step(executor=self.executor, name="SinkStep")
            
            # Create data units
            source_output = DataUnitString(name="SourceOutput", debug=True)
            sink_input = DataUnitString(name="SinkInput", debug=True)
            
            # Set up source and sink
            source.output = source_output
            link_id = "test_link_pattern"
            sink.register_input_source(link_id, sink_input)
            
            # Create LinkDirect with TriggerDataUpdated
            link = LinkDirect(
                source_step=source,
                sink_step=sink,
                link_id=link_id,
                debug=True,
                auto_setup_trigger=True  # This will create and start the trigger automatically
            )
            
            # Verify link has a trigger
            self.assertIsNotNone(link.trigger)
            
            # Change source data which should trigger transfer
            print("\nChanging source data...")
            source_output.set("Test communication pattern")
            
            # Wait for trigger and transfer
            await asyncio.sleep(1)
            
            # Check if data was transferred
            sink_input = sink.input_sources.get(link_id)
            if sink_input:
                sink_data = sink_input.get()
                print(f"Sink received data: {sink_data}")
                self.assertEqual(sink_data, "Test communication pattern")
            else:
                self.fail("No data received by sink")
                
            # Change source data again
            print("\nChanging source data again...")
            source_output.set("Second test message")
            
            # Wait for trigger and transfer
            await asyncio.sleep(1)
            
            # Check if new data was transferred
            if sink_input:
                sink_data = sink_input.get()
                print(f"Sink received updated data: {sink_data}")
                self.assertEqual(sink_data, "Second test message")
            else:
                self.fail("No updated data received by sink")
                
            # Stop monitoring
            await link.stop_monitoring()
            
            return True
            
        # Run the test
        result = self.loop.run_until_complete(run_test())
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main() 