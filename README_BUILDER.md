# NanoBrain Builder

The NanoBrain Builder is a command-line tool for creating and managing NanoBrain workflows. It provides a structured approach to building complex neural-inspired processing systems.

## Features

- **Workflow Creation**: Create new workflows with proper directory structure and configuration
- **Step Management**: Create, test, and save processing steps
- **Link Creation**: Connect steps together to form a complete workflow
- **AI-Powered Tools**: Use AI to help plan, code, and document your workflows
- **Git Integration**: Initialize and manage git repositories for your workflows

## Installation

The NanoBrain Builder is included as part of the NanoBrain framework. To use it, simply clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/nanobrain.git
cd nanobrain
pip install -r requirements.txt
```

## Quick Start

### Create a New Workflow

```bash
./nanobrain_builder.py create-workflow my_workflow
```

This will create a new workflow directory with the following structure:

```
my_workflow/
├── config/
├── src/
└── test/
```

### Create a Step

```bash
./nanobrain_builder.py create-step data_processing
```

This will create a new step in the current workflow:

```
my_workflow/
├── config/
├── src/
│   └── StepDataProcessing/
│       ├── __init__.py
│       ├── StepDataProcessing.py
│       └── config/
│           └── StepDataProcessing.yml
└── test/
    └── test_StepDataProcessing.py
```

### Link Steps Together

```bash
./nanobrain_builder.py link-steps data_processing visualization
```

This will create a link between the two steps:

```
my_workflow/
├── config/
│   └── StepDataProcessingToStepVisualizationLink.yml
├── src/
│   ├── StepDataProcessing/
│   ├── StepVisualization/
│   └── StepDataProcessingToStepVisualizationLink.py
└── test/
```

### Save the Workflow

```bash
./nanobrain_builder.py save-workflow
```

This will finalize the workflow, ensuring all steps are properly linked and tested.

## Documentation

For more detailed documentation, see the [builder documentation](builder/README.md).

## Architecture

The builder system is divided into several components:

1. **NanoBrainBuilder** - The main class that provides the high-level API for creating and managing workflows.
2. **WorkflowSteps** - A collection of step classes that implement the functionality for each command.
3. **Agent** - An AI-powered agent that helps with various tasks such as generating code and planning.
4. **Tools** - A collection of tools used by the agent to perform specific tasks.

### NanoBrainBuilder

The `NanoBrainBuilder` class is the main entry point for the builder. It provides methods for creating and managing workflows, steps, and links. It also maintains a stack of active workflows, allowing you to work on multiple workflows at once.

### WorkflowSteps

The `WorkflowSteps` module contains the implementation of the workflow steps. Each step is represented by a class with a static `execute` method that performs the step's functionality. The following steps are available:

- `CreateWorkflowStep` - Creates a new workflow directory with the necessary files and structure.
- `CreateStepStep` - Creates a new step within a workflow.
- `TestStepStep` - Tests a step by running its unit tests.
- `SaveStepStep` - Finalizes a step after testing.
- `LinkStepsStep` - Creates a link between two steps.
- `SaveWorkflowStep` - Finalizes a workflow after all steps are linked and tested.

### Agent

The `Agent` class is an AI-powered assistant that helps with various tasks. It uses a language model to generate code, plan step implementations, and provide guidance. The agent can be configured with different language models and prompt templates.

### Tools

The builder uses several tools to perform specific tasks:

- `StepFileWriter` - Writes files to disk.
- `StepPlanner` - Plans the implementation of a step.
- `StepCoder` - Generates code for a step.
- `StepGitInit` - Initializes a git repository.
- `StepContextSearch` - Searches for context in the codebase.
- `StepWebSearch` - Searches the web for information.

## Testing

The builder includes a comprehensive test suite to ensure that it works correctly. The tests are divided into three categories:

1. **Simple Tests** - Basic tests that verify the functionality of the builder's methods without requiring the actual implementation.
2. **Actual Tests** - Tests that verify the actual implementation of the builder.
3. **Workflow Execution Tests** - Tests that verify that workflows created by the builder can be executed correctly.

To run the tests, use the following command:

```bash
./test/run_builder_tests.py
```

### Test Structure

The test files are located in the `test` directory:

- `test_builder_simple.py` - Simple tests for the builder.
- `test_builder_actual.py` - Tests for the actual implementation of the builder.
- `test_workflow_execution.py` - Tests for workflow execution.
- `run_builder_tests.py` - Script to run all tests.
- `mock_agent.py` - Mock implementation of the Agent class.
- `mock_builder.py` - Mock implementation of the NanoBrainBuilder class.
- `mock_executor.py` - Mock implementation of the ExecutorBase class.
- `mock_tools.py` - Mock implementations of the various tools.

## Biological Inspiration

The NanoBrain Builder is inspired by the brain's prefrontal cortex, which is responsible for planning, decision-making, and coordinating complex cognitive behaviors. Like the prefrontal cortex, the builder helps coordinate the creation of complex processing systems by breaking them down into manageable steps and ensuring they work together correctly.

## Contributing

Contributions to the NanoBrain Builder are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information. 