"""
Custom prompt template for tool calling.

This module provides a template for tool calling that ensures the LLM passes
the correct arguments and uses tools in the appropriate context.
"""

TOOL_CALLING_SYSTEM_PROMPT = """
You are an AI assistant with access to a set of tools. When you need to use a tool, use the following format:

<tool>
name: [tool name]
args: [comma-separated list of arguments]
</tool>

For example:
<tool>
name: CalculatorStep
args: add, 5, 3
</tool>

Available tools:
{tool_descriptions}

Remember to:
1. Only use tools that are listed above
2. Use the exact tool name as provided
3. Provide arguments in the correct order and format
4. Only use tools when necessary
5. If you don't need to use a tool, just respond normally
6. If request includes creating or writing code, use corresponding tools to produce it and save files

Context from previous conversation:
{context}

User: {input}
Assistant: 
"""

def create_tool_calling_prompt(tools):
    """
    Create a prompt template for tool calling.
    
    Args:
        tools: List of Step objects to be used as tools
        
    Returns:
        A formatted system prompt with tool descriptions
    """
    tool_descriptions = []
    
    for tool in tools:
        # Get the tool name
        tool_name = tool.__class__.__name__
        
        # Get the tool description from its docstring
        tool_description = tool.__doc__ or f"Execute the {tool_name} step"
        
        # Get the process method's docstring for argument information
        process_doc = tool.process.__doc__ or ""
        
        # Format the tool description
        formatted_description = f"""
- {tool_name}:
  Description: {tool_description.strip()}
  Usage: {process_doc.strip()}
"""
        tool_descriptions.append(formatted_description)
    
    # Join all tool descriptions
    all_descriptions = "\n".join(tool_descriptions)
    
    # Return the formatted prompt
    return TOOL_CALLING_SYSTEM_PROMPT.format(tool_descriptions=all_descriptions, context="{context}", input="{input}")

def parse_tool_call(response):
    """
    Parse a tool call from the LLM response.
    
    Args:
        response: The LLM response text
        
    Returns:
        A tuple of (tool_name, args) if a tool call is found, None otherwise
    """
    import re
    
    # Define the pattern to match tool calls
    pattern = r'<tool>\s*name:\s*([^\n]+)\s*args:\s*([^\n]+)\s*</tool>'
    
    # Search for the pattern in the response
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        
        # Split the args by comma and strip whitespace
        args = [arg.strip() for arg in args_str.split(',')]
        
        return (tool_name, args)
    
    return None 