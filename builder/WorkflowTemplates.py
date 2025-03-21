#!/usr/bin/env python3
"""
WorkflowTemplates - Templates for creating NanoBrain components.

This module provides templates for creating and managing
workflows, steps, links, and other components in the NanoBrain framework.
These templates are used by the workflow builder functionality.
"""

# Step Templates
STEP_TEMPLATE = '''#!/usr/bin/env python3
"""
{step_class_name} - {description}

This step implements {description} for NanoBrain workflows.
"""

from src.{base_class} import {base_class}


class {step_class_name}({base_class}):
    """
    {description}
    
    Biological analogy: Specialized neuron.
    Justification: Like how specialized neurons perform specific functions
    in the brain, this step performs a specific function in the workflow.
    """
    
    def __init__(self, **kwargs):
        """Initialize the step."""
        super().__init__(**kwargs)
        
    def process(self, data_dict):
        """
        Process input data.
        
        Args:
            data_dict: Dictionary containing input data
            
        Returns:
            Dictionary containing output data
        """
        # Process the input data
        result = {{}}
        
        # Add your custom processing logic here
        # ...
        
        return result
'''

STEP_CONFIG_TEMPLATE = '''# Default configuration for {step_class_name}
defaults:
  # Add your default configuration parameters here
  debug_mode: false
  monitoring: true

  # Step-specific configuration
  name: "{step_class_name}"
  description: "{description}"
'''

STEP_INIT_TEMPLATE = '''from .{step_class_name} import {step_class_name}

__all__ = ['{step_class_name}']
'''

# Workflow Templates
WORKFLOW_TEMPLATE = '''from typing import List, Dict, Any, Optional
from src.Workflow import Workflow
from src.ExecutorBase import ExecutorBase
from src.Step import Step

class {workflow_class_name}(Workflow):
    """
    {workflow_name} workflow.
    
    Biological analogy: Coordinated neural circuit.
    Justification: Like how coordinated neural circuits work together to
    accomplish complex tasks, this workflow coordinates multiple steps
    to achieve its objectives.
    """
    def __init__(self, executor: Optional[ExecutorBase] = None, steps: List[Step] = None, **kwargs):
        # Create an executor if none is provided
        if executor is None:
            from src.ExecutorBase import ExecutorBase
            executor = ExecutorBase()
        
        # Initialize the Workflow base class
        super().__init__(executor, steps, **kwargs)
        
        # {workflow_class_name}-specific attributes
        # Add your attributes here
'''

WORKFLOW_CONFIG_TEMPLATE = '''defaults:
  # Add your default configuration parameters here

metadata:
  description: "{workflow_name} workflow"
  biological_analogy: "Coordinated neural circuit"
  justification: >
    Like how coordinated neural circuits work together to accomplish complex tasks,
    this workflow coordinates multiple steps to achieve its objectives.
  objectives:
    # Add your workflow objectives here
  author: "{author_name}"

validation:
  required:
    - executor  # ExecutorBase instance required
  optional:
    # Add your optional parameters here
  constraints:
    # Add your parameter constraints here

examples:
  basic:
    description: "Basic usage example"
    config:
      # Add example configuration here
'''

WORKFLOW_README_TEMPLATE = '''# {workflow_name}

A NanoBrain workflow created with the NanoBrain builder tool.

## Author
{author_name}
'''

# Link Templates
LINK_TEMPLATE = '''from typing import Any
from src.{link_type} import {link_type}
from src.{source_class_name}.{source_class_name} import {source_class_name}
from src.{target_class_name}.{target_class_name} import {target_class_name}

class {link_name}({link_type}):
    """
    Link from {source_class_name} to {target_class_name}.
    
    Biological analogy: Synaptic connection.
    Justification: Like how synaptic connections transmit signals between
    neurons, this link transmits data from {source_class_name} to {target_class_name}.
    """
    def __init__(self, source_step: {source_class_name}, target_step: {target_class_name}, **kwargs):
        # Initialize with data units from the steps
        super().__init__(
            input_data=source_step.output_data,
            output_data=target_step.input_data,
            **kwargs
        )
        
        # Store references to the steps
        self.source_step = source_step
        self.target_step = target_step
    
    async def transfer(self) -> Any:
        """
        Transfer data from the source step to the target step.
        
        Returns:
            The transferred data
        """
        # Use the base class transfer method
        return await super().transfer()
'''

LINK_CONFIG_TEMPLATE = '''defaults:
  # Add your default configuration parameters here

metadata:
  description: "Link from {source_class_name} to {target_class_name}"
  biological_analogy: "Synaptic connection"
  justification: >
    Like how synaptic connections transmit signals between neurons,
    this link transmits data from {source_class_name} to {target_class_name}.
  objectives:
    - Transfer data from {source_class_name} to {target_class_name}
    - Ensure reliable data transmission

validation:
  required:
    - input_data  # DataUnitBase instance required
    - output_data  # DataUnitBase instance required
  optional:
    # Add your optional parameters here
  constraints:
    # Add your parameter constraints here

examples:
  basic:
    description: "Basic usage example"
    config:
      # Add example configuration here
''' 