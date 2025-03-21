#!/usr/bin/env python3
"""
CommandLinePrompts - Templates for command line prompts in NanoBrain.

This module provides prompt templates for generating code, configuration files,
and other content for NanoBrain components. These are used by the different
builder tools and steps to interact with the LLM.
"""

# Initial template generation prompts
TEMPLATE_CODE_PROMPT = """
Please generate an initial template for a step class with the following details:
- Class name: {step_class_name}
- Base class: {base_class}
- Description: {description}

The template should include:
- Proper imports
- Class docstring with biological analogy
- Constructor with proper initialization
- A basic process method
- Any needed helper methods

Return the complete code file that I can save as {step_class_name}.py.
"""

TEMPLATE_CONFIG_PROMPT = """
Please generate a YAML configuration file for the {step_class_name} class with the following details:
- Class name: {step_class_name}
- Description: {description}

The configuration should include:
- Default parameters
- Metadata section with description
- Other relevant YAML configuration elements

Return the complete YAML content that I can save as {step_class_name}.yml.
"""

# Final configuration generation prompt
FINAL_CONFIG_PROMPT = """
Please generate a final YAML configuration file for the {step_class_name} class with the following details:
- Class name: {step_class_name}
- Step location: {step_dir}

The configuration should include:
- Default parameters based on the implemented step
- Metadata section with description
- Any parameters used in the implementation

Return the complete YAML content that I can save as {step_class_name}.yml.
"""

# Integration and code generation prompts
IMPLEMENTATION_INTEGRATION_PROMPT = """
I need to integrate solution code into a Step class implementation.

The current Step implementation is in the file {step_file_path}:

```python
{step_code}
```

I have a solution file with code that needs to be integrated:

```python
{solution_code}
```

Please:
1. Merge the solution code into the Step class's process method
2. Keep the class structure intact
3. Add necessary imports
4. Make sure the code is well-organized and follows best practices

Save the updated implementation to {step_file_path}.
"""

# Test update prompt
TEST_UPDATE_PROMPT = """
I need to update the test file for the Step class to match the new implementation.

The Step class is at {step_file_path}. 

The test file is located at {test_file_path} with the current implementation:

```python
{test_code}
```

Please:
1. Update the test cases to cover the functionality of the updated implementation
2. Update mocks and test data as needed
3. Keep the test structure consistent with existing tests

Save the updated test implementation to {test_file_path}.
"""

# Command line help text
COMMAND_LINE_HELP_TEXT = """
Available commands:
1. link <source_step> <target_step> [link_type] - Link this step to another step
2. finish - End step creation and save
3. help - Show this menu
Other inputs will be used to enhance the step's code. Examples:
- "Add a method to process JSON data"
- "Implement error handling for network requests"
- "The step should validate input parameters"
"""

# Welcome messages
STEP_CREATION_WELCOME = """
🎉 Step Creation Wizard
====================
📁 Creating step '{step_class_name}' in workflow '{workflow_path}'

💡 Instructions:
1. Describe the problem you want to solve with this step
2. The AI will generate a solution based on your description
3. Review the solution and provide feedback or additional requirements
4. Type 'finish' when you're satisfied to integrate the solution into your Step class
5. Type 'help' for more commands

▶️ Begin by describing what this step should do:
""" 