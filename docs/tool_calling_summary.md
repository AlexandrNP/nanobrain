# Tool Calling Implementation Summary

## Overview

We have successfully implemented tool calling capability for the `Agent` class, which allows it to use other `Step` objects as tools to perform specific tasks. This implementation provides two approaches:

1. **LangChain Tool Binding**: Wrapping each `Step` class as a LangChain Tool and registering it with the LLM via the `bind_tools` function.
2. **Custom Tool Calling Prompt**: Using a custom-written tool calling prompt that ensures passing the correct arguments and using the tool in the appropriate context.

## Implementation Details

### Agent Class Modifications

1. Added support for registering tools in the `Agent` constructor:
   ```python
   def __init__(self, 
                executor: ExecutorBase,
                model_name: str = "gpt-3.5-turbo",
                model_class: Optional[str] = None,
                memory_window_size: int = 5,
                prompt_file: str = None,
                prompt_template: str = None,
                prompt_variables: Optional[Dict] = None,
                use_shared_context: bool = False,
                shared_context_key: Optional[str] = None,
                tools: Optional[List[Step]] = None,
                use_custom_tool_prompt: bool = False,
                **kwargs):
   ```

2. Implemented the `_register_tools` method to register `Step` objects as tools for the LLM:
   ```python
   def _register_tools(self, tools: List[Step]):
       """
       Register Step objects as tools for the LLM.
       """
       self.langchain_tools = []
       
       for step in tools:
           # Create a tool from the Step object
           step_tool = self._create_tool_from_step(step)
           if step_tool:
               self.langchain_tools.append(step_tool)
       
       # Bind tools to the LLM if we have any and if the LLM supports tool binding
       if self.langchain_tools and hasattr(self.llm, 'bind_tools'):
           self.llm = self.llm.bind_tools(self.langchain_tools)
   ```

3. Implemented the `_create_tool_from_step` method to create a LangChain tool from a `Step` object:
   ```python
   def _create_tool_from_step(self, step: Step) -> Optional[BaseTool]:
       """
       Create a LangChain tool from a Step object.
       """
       # Implementation details...
   ```

4. Added methods for adding and removing tools:
   ```python
   def add_tool(self, step: Step):
       """
       Add a new tool to the agent.
       """
       # Implementation details...
   
   def remove_tool(self, step: Step):
       """
       Remove a tool from the agent.
       """
       # Implementation details...
   ```

5. Implemented the `process_with_tools` method to process inputs using the language model with tool calling capability:
   ```python
   async def process_with_tools(self, inputs: List[Any]) -> Any:
       """
       Process inputs using the language model with tool calling capability.
       """
       # Implementation details...
   ```

6. Added the `execute_tool` method to execute a specific tool by name:
   ```python
   async def execute_tool(self, tool_name: str, args: List[Any]) -> Any:
       """
       Execute a specific tool by name with the given arguments.
       """
       # Implementation details...
   ```

### Custom Tool Calling Prompt

1. Created a custom prompt template for tool calling in `prompts/tool_calling_prompt.py`:
   ```python
   TOOL_CALLING_SYSTEM_PROMPT = """
   You are an AI assistant with access to a set of tools. When you need to use a tool, use the following format:
   
   <tool>
   name: [tool name]
   args: [comma-separated list of arguments]
   </tool>
   
   # ... more details ...
   """
   ```

2. Implemented functions to create the prompt and parse tool calls:
   ```python
   def create_tool_calling_prompt(tools):
       """
       Create a prompt template for tool calling.
       """
       # Implementation details...
   
   def parse_tool_call(response):
       """
       Parse a tool call from the LLM response.
       """
       # Implementation details...
   ```

### Mock Implementations for Testing

1. Added mock implementations of LangChain classes in `mock_langchain.py`:
   ```python
   class MockChatOpenAI:
       """Mock implementation of ChatOpenAI."""
       # Implementation details...
   
   class MockOpenAI:
       """Mock implementation of OpenAI."""
       # Implementation details...
   
   class MockBaseTool:
       """Mock implementation of BaseTool."""
       # Implementation details...
   
   def tool(func=None, name=None, description=None, **kwargs):
       """Mock implementation of the tool decorator."""
       # Implementation details...
   ```

## Testing

1. Created test cases for the LangChain tool binding approach in `test/test_agent_tools.py`:
   - `test_tool_registration`: Tests that tools are properly registered with the agent.
   - `test_add_tool`: Tests adding a tool to the agent.
   - `test_remove_tool`: Tests removing a tool from the agent.
   - `test_tool_creation`: Tests the creation of a tool from a Step object.
   - `test_process_with_tools`: Tests processing with tools.

2. Created test cases for the custom tool prompt approach in `test/test_agent_custom_tool_prompt.py`:
   - `test_parse_tool_call`: Tests parsing tool calls from LLM responses.
   - `test_process_with_custom_prompt`: Tests processing with custom tool prompt.
   - `test_execute_tool_directly`: Tests executing a tool directly by name.

## Documentation

1. Created documentation for the tool calling capability in `docs/tool_calling.md`:
   - Overview of the tool calling capability
   - Instructions for using LangChain tool binding
   - Instructions for using custom tool calling prompt
   - Instructions for creating tool steps
   - Comparison of the two approaches
   - Biological analogy for the tool calling capability
   - Testing instructions

## Conclusion

The tool calling capability for the `Agent` class has been successfully implemented and tested. This capability allows the `Agent` to use other `Step` objects as tools to perform specific tasks, extending its problem-solving capabilities. The implementation provides two approaches, giving flexibility in how tool calling is used in different contexts. 