"""
DataStorageCommandLine Module

This module provides a command-line interface for data storage and interaction.
"""

from typing import Any, List, Optional, Dict, Union, Callable
import asyncio
import sys
import os
from datetime import datetime
import traceback
"""
DataStorageCommandLine Module

This module provides a command-line interface for data storage and interaction.
"""

from typing import Any, List, Optional, Dict, Union, Callable
import asyncio
import sys
import os
from datetime import datetime
import traceback
import time
import copy

from src.DataStorageBase import DataStorageBase
from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase
from src.TriggerBase import TriggerBase
from src.enums import ComponentState
from src.Step import Step


class DataStorageCommandLine(Step):
    """
    Command line interface for data storage and retrieval.
    
    Biological analogy: Sensory input system.
    Justification: Like how sensory systems gather external information and pass it to
    the brain for processing, the command line gathers user input and passes it to
    the system for processing.
    """
    
    def __init__(self, name="CommandLine", history_size=20, **kwargs):
        """
        Initialize the command line interface.
        
        Args:
            name: Name of the interface
            history_size: Maximum number of history entries to keep
        """
        super().__init__(name=name, **kwargs)
        self.history: List[Dict[str, str]] = []  # Store interaction history
        self.history_size = history_size  # Maximum size of history
        self.prompt = kwargs.get('prompt', '> ')  # Prompt for user input
        self.welcome_message = kwargs.get('welcome_message', 'Welcome to the command line interface.')
        self.goodbye_message = kwargs.get('goodbye_message', 'Goodbye!')
        self.exit_command = kwargs.get('exit_command', 'exit')
        self._monitoring = False
        self._monitor_task = None
        self.agent_builder = None  # Direct reference to agent builder if available
        self._debug_mode = kwargs.get('debug', False)  # Enable debug output
        
    def _force_output_change(self, data):
        """
        Force an output change to ensure it's detected by triggers.
        
        Biological analogy: Neurotransmitter release.
        Justification: Like how neurons forcefully release neurotransmitters
        to ensure signal transmission, this method forcefully updates the output
        to ensure the change is detected.
        
        Args:
            data: The data to set as output
            
        Returns:
            True if successful, False otherwise
        """
        if not hasattr(self, 'output') or not self.output:
            if self._debug_mode:
                print("DataStorageCommandLine: No output to update")
            return False
            
        # Create a deep copy to ensure it's seen as a new object
        copied_data = copy.deepcopy(data)
        
        # Force update with a new object instance
        self.output.set(copied_data)
        
        if self._debug_mode:
            print(f"DataStorageCommandLine: Forced output change to: {copied_data}")
            
        # If we have a direct reference to agent_builder, trigger it directly too
        if self.agent_builder and hasattr(self.agent_builder, 'process'):
            if self._debug_mode:
                print("DataStorageCommandLine: Also triggering agent_builder directly")
            asyncio.create_task(self.agent_builder.process([copied_data]))
            
        return True
        
    async def process(self, inputs=None):
        """
        Process input data and produce output.
        
        Biological analogy: Memory retrieval and encoding.
        Justification: Like how the brain retrieves stored information and encodes
        new information, this method retrieves input data and encodes output data.
        
        Args:
            inputs: Input data to process
            
        Returns:
            Processed data
        """
        if not inputs:
            if self._debug_mode:
                print("DataStorageCommandLine: No inputs to process")
            return None
            
        # Extract query from inputs
        query = None
        if isinstance(inputs, list) and len(inputs) > 0:
            query = inputs[0]
        elif isinstance(inputs, dict) and 'query' in inputs:
            query = inputs['query']
        elif isinstance(inputs, str):
            query = inputs
            
        # Check if query is provided
        if not query:
            if self._debug_mode:
                print("DataStorageCommandLine: No query provided")
            return None
            
        # Process the query
        response = self._process_query(query)
        
        # Update history with query and response
        self._add_to_history(query, response)
        
        # Use the force_output_change method to ensure change detection
        if response is not None:
            self._force_output_change(response)
        
        return response
        
    def _process_query(self, query: str) -> str:
        """
        Process a query string.
        
        Biological analogy: Language processing.
        Justification: Like how the brain processes language input to extract meaning,
        this method processes query strings to determine the response.
        
        Args:
            query: Query string to process
            
        Returns:
            Response string
        """
        # Check for specific command pattern: <class_name>><instructions>
        import re
        class_pattern = re.match(r'<([A-Za-z0-9_]+)>>(.+)', query)
        if class_pattern:
            class_name = class_pattern.group(1)
            instructions = class_pattern.group(2).strip()
            
            if self._debug_mode:
                print(f"DataStorageCommandLine: Detected class-specific command pattern")
                print(f"  Class: {class_name}")
                print(f"  Instructions: {instructions}")
                
            # Format a prompt specifically for generating class code
            return f"Generate a class named {class_name} with the following requirements:\n{instructions}"
            
        # In the simple implementation, just return the query
        return query
        
    def _add_to_history(self, query: str, response: str) -> None:
        """
        Add a query-response pair to history.
        
        Biological analogy: Memory formation.
        Justification: Like how the brain forms memories by encoding and storing experiences,
        this method stores interaction history for later retrieval.
        
        Args:
            query: Query string
            response: Response string
        """
        self.history.append({
            'query': query,
            'response': response,
            'timestamp': time.time()
        })
        
        # Keep history size limited
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
            
    def get_history(self) -> List[Dict[str, Union[str, float]]]:
        """
        Get interaction history.
        
        Biological analogy: Memory retrieval.
        Justification: Like how the brain retrieves stored memories,
        this method retrieves stored interaction history.
        
        Returns:
            List of history entries
        """
        return self.history
        
    async def start_monitoring(self) -> None:
        """
        Start monitoring for user input.
        
        Biological analogy: Sensory attention.
        Justification: Like how sensory systems attend to external stimuli,
        this method attends to user input.
        """
        if self._monitoring:
            return  # Already monitoring
            
        self._monitoring = True
        
        # Show welcome message
        print(self.welcome_message)
        
        # Show instructions for the user
        print("\n💡 You are now in an interactive step creation session.")
        print("💡 Describe what you want this step to do, and I'll create the code for you.")
        print("💡 You can use the following commands:")
        print("   - Type 'help' to see available commands")
        print("   - Type 'finish' when you're done creating the step")
        print("   - Type 'link <source_step> <target_step>' to link steps")
        print("   - Any other input will be used to enhance the step's code\n")
        
        # Show data flow information if debug mode is enabled
        if self._debug_mode:
            print("\nData Flow Information:")
            print("1. User input -> DataStorageCommandLine.process")
            print("2. DataStorageCommandLine.process -> _force_output_change")
            print("3. _force_output_change -> output.set")
            print("4. TriggerDataUpdated detects output change")
            print("5. TriggerDataUpdated -> LinkDirect.transfer")
            print("6. LinkDirect.transfer -> AgentWorkflowBuilder.process")
            
            # Show connection information
            if hasattr(self, 'agent_builder') and self.agent_builder:
                print("\nDirect Connection: CommandLine -> AgentBuilder (backup path)")
            else:
                print("\nNo direct connection. Using Link/Trigger mechanism only.")
                
            # Show output information
            if hasattr(self, 'output') and self.output:
                print(f"Output Data Unit: {self.output.__class__.__name__}")
            else:
                print("No output data unit configured.")
        
        try:
            # Loop to get user input
            while self._monitoring:
                # Show prompt and get input
                sys.stdout.write(self.prompt)
                sys.stdout.flush()
                
                # Get user input
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(None, input)
                except EOFError:
                    # Handle Ctrl+D
                    print("\nEOF detected. Exiting.")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print("\nInterrupt detected. Exiting.")
                    break
                
                # Check for exit command
                if user_input.strip().lower() == self.exit_command:
                    print(self.goodbye_message)
                    break
                
                try:
                    # Handle special commands directly here to avoid async issues
                    if user_input.strip().lower() == "finish":
                        # Handle finish command
                        print("✅ Finishing step creation...")
                        
                        # Stop monitoring to exit the loop
                        self._monitoring = False
                        response = "Step creation completed."
                        self._add_to_history(user_input, response)
                        self._force_output_change(response)
                        break
                    elif user_input.strip().lower() == "help":
                        # Handle help command
                        help_text = """
Available commands:
1. link <source_step> <target_step> [link_type] - Link this step to another step
2. finish - End step creation and save
3. help - Show this menu

Other inputs will be used to enhance the step's code. Examples:
- "Add a method to process JSON data"
- "Implement error handling for network requests"
- "The step should validate input parameters"
"""
                        print(help_text)
                        
                        self._add_to_history(user_input, help_text)
                        self._force_output_change(help_text)
                        continue
                    elif user_input.strip().lower().startswith("link "):
                        # Handle link command: link <source_step> <target_step>
                        parts = user_input.strip().split()
                        if len(parts) >= 3:
                            source_step = parts[1]
                            target_step = parts[2]
                            link_type = parts[3] if len(parts) > 3 else "LinkDirect"
                            
                            print(f"🔗 Linking steps: {source_step} -> {target_step} using {link_type}...")
                            
                            response = f"Linking steps: {source_step} -> {target_step} using {link_type}"
                            self._add_to_history(user_input, response)
                            self._force_output_change(response)
                            continue
                    
                    # For all other inputs, use the process method
                    print("⚙️ Processing your input and updating the step code...")
                    result = await self.process(user_input)
                    
                    if result:
                        print(f"\n✅ Code updated based on your input. Keep adding more details or type 'finish' when done.")
                    else:
                        print(f"\n⚠️ No changes were made. Please provide more specific instructions.")
                    
                    if self._debug_mode:
                        print(f"Process result: {result}")
                        
                        # Check if output was updated
                        if hasattr(self, 'output') and self.output:
                            print(f"Current output value: {self.output.get()}")
                    
                except asyncio.CancelledError:
                    # Handle cancellation
                    break
                except Exception as e:
                    # Handle other exceptions
                    print(f"❌ Error processing input: {e}")
                    traceback.print_exc()
        
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        finally:
            # Clean up
            self._monitoring = False
            print("\n✅ Step creation session ended. Files will be saved.")
            
    def stop_monitoring(self) -> None:
        """
        Stop monitoring for user input.
        
        Biological analogy: Sensory inhibition.
        Justification: Like how sensory systems can inhibit attention to stimuli,
        this method stops attending to user input.
        """
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None 