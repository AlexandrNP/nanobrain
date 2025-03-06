# NanoBrain Builder

The NanoBrain Builder is a command-line interface for building NanoBrain workflows.

## Overview

The builder uses a combination of AI-powered tools and a step-by-step approach to create, test, and manage NanoBrain workflows. It guides you through the process of building a workflow, creating steps, linking them together, and testing the result.

## Biological Analogy

The NanoBrain Builder is analogous to the brain's prefrontal cortex executive function:

- **Prefrontal Cortex**: Like how the prefrontal cortex handles planning and decision-making, the NanoBrainBuilder plans and constructs workflows, managing the various steps and connections between them.

## Usage

### Command-Line Interface

The builder can be used as a command-line tool:

```
python nanobrain_builder.py <command> [options]
```

### Available Commands

- `create-workflow <name>`: Create a new workflow
- `create-step <name>`: Create a new step in the current workflow
- `test-step <name>`: Test a step in the current workflow
- `save-step <name>`: Save a step after testing
- `link-steps <source> <target>`: Link two steps together
- `save-workflow`: Save the current workflow

### Examples

Create a new workflow:

```
python nanobrain_builder.py create-workflow my_workflow
```

Create a step within the workflow:

```
python nanobrain_builder.py create-step data_processing
```

Link two steps together:

```
python nanobrain_builder.py link-steps data_processing visualization
```

## Architecture

The builder consists of several components:

- **NanoBrainBuilder**: Main class for interacting with the builder
- **WorkflowSteps**: Implementation of the workflow steps (create-workflow, create-step, etc.)
- **Agent**: AI-powered agent for handling complex tasks
- **Tools**: Various tools used by the agent (file writing, planning, coding, etc.)

### Implementation Details

The builder is implemented using the following classes:

1. **NanoBrainBuilder**: The main class that provides methods for creating workflows, steps, and links. It uses an Agent to perform complex tasks and maintains a stack of active workflows.

2. **WorkflowSteps**: A collection of step classes that implement the actual workflow operations:
   - `CreateWorkflowStep`: Creates a new workflow directory and initializes it
   - `CreateStepStep`: Creates a new step within a workflow
   - `TestStepStep`: Tests a step by running its unit tests
   - `SaveStepStep`: Finalizes a step after testing
   - `LinkStepsStep`: Creates a link between two steps
   - `SaveWorkflowStep`: Finalizes a workflow after all steps are linked and tested

3. **Tools**: The builder uses several tools to perform specific tasks:
   - `StepFileWriter`: Writes files to disk
   - `StepPlanner`: Plans the implementation of a step
   - `StepCoder`: Generates code for a step
   - `StepGitInit`: Initializes a git repository
   - `StepContextSearch`: Searches for context in the codebase
   - `StepWebSearch`: Searches the web for information

## Example Workflow

A typical workflow creation process using the builder would look like this:

1. Create a new workflow
2. Create multiple steps (input, processing, output, etc.)
3. Test each step individually
4. Save each step after testing
5. Link the steps together
6. Save the workflow

## Extending the Builder

The builder can be extended by adding new tools to the agent or new workflow steps. To add a new tool:

1. Create a new tool class that inherits from `Step`
2. Implement the `process` method to perform the tool's task
3. Add the tool to the agent in the `_init_tools` method of `NanoBrainBuilder`

To add a new workflow step:

1. Create a new step class in `WorkflowSteps.py`
2. Implement the `execute` method to perform the step's task
3. Add a method to `NanoBrainBuilder` that calls the step's `execute` method
4. Update the command-line interface to include the new step 