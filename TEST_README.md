# Nanobrain Test Documentation

This document provides a comprehensive overview of the Nanobrain project's test suite, describing test classes, their purposes, test functions, edge cases considered, and areas for improvement.

## Table of Contents

1. [Test Organization](#test-organization)
2. [Core Component Tests (`test/`)](#core-component-tests)
3. [Builder Tests (`builder/`)](#builder-tests)
4. [Integration Tests (`integration_tests/`)](#integration-tests)
5. [Mock Components](#mock-components)
6. [Test Infrastructure](#test-infrastructure)
7. [Areas for Improvement](#areas-for-improvement)

## Test Organization

The test suite is organized into three main categories:

1. **Core Component Tests** (`test/`): Unit tests for individual components of the Nanobrain framework
2. **Builder Tests** (`builder/`): Tests specific to the builder functionality
3. **Integration Tests** (`integration_tests/`): End-to-end tests of multiple components working together

## Core Component Tests

### Link and Trigger Tests

#### `test_link_refactoring.py`

**Purpose:** Tests the refactored Link class hierarchy and interaction with trigger types.

**Test Classes:**
- `TestLinkRefactoring`: Verifies LinkBase class compatibility with different trigger types.

**Key Test Functions:**
- `test_link_base_with_auto_trigger_data_changed`: Tests auto-creation of TriggerDataUpdated
- `test_link_base_with_auto_trigger_data_hash_changed`: Tests auto-creation of TriggerDataHashChanged
- `test_link_base_with_manual_trigger`: Tests using manually provided triggers
- `test_link_direct_with_manual_trigger`: Tests LinkDirect with manually provided trigger
- `test_link_file_with_manual_trigger`: Tests LinkFile with manually provided trigger
- `test_link_base_with_no_trigger`: Tests LinkBase behavior without a trigger
- `test_data_deduplication`: Tests prevention of duplicate data transfers

**Edge Cases:**
- Null trigger with auto_setup_trigger=False
- Data duplication prevention when same data is sent multiple times
- Testing both TriggerDataUpdated and TriggerDataHashChanged behavior

#### `test_link_trigger_refactor.py`

**Purpose:** Tests the interaction between Link classes and different trigger types.

**Test Classes:**
- `TestLinkRefactor`: Tests behavior of Link classes with different trigger implementations.

**Key Test Functions:**
- `test_linkbase_data_changed_trigger`: Tests LinkBase with TriggerDataUpdated
- `test_linkbase_data_hash_changed_trigger`: Tests LinkBase with TriggerDataHashChanged
- `test_linkdirect_with_data_changed_trigger`: Tests LinkDirect with TriggerDataUpdated
- `test_linkdirect_with_hash_changed_trigger`: Tests LinkDirect with TriggerDataHashChanged
- `test_duplicate_transfer_prevention`: Tests prevention of duplicate data transfers
- `test_monitoring_activation`: Tests activation and deactivation of monitoring

**Edge Cases:**
- Starting and stopping monitoring
- Manual triggering of monitor/transfer functions
- Process synchronization with sync/async operations
- Data transfer validation

### Agent Tests

#### `test_agent.py`

**Purpose:** Tests the Agent class functionality for interacting with LLMs.

**Test Classes:**
- `TestAgent`: Tests Agent initialization, LLM loading, and querying.

**Key Test Functions:**
- `test_initialization`: Verifies Agent object structure
- `test_initialize_llm`: Tests LLM initialization with different models
- `test_load_prompt_template`: Tests prompt template loading
- `test_load_prompt_template_from_file`: Tests prompt loading from files
- `test_process`: Tests asynchronous prompt processing with LLM

**Edge Cases:**
- Missing API keys
- Custom prompt templates
- Various model configurations

#### `test_agent_memory.py`

**Purpose:** Tests Agent's memory management for conversation history.

**Test Classes:**
- `TestAgentMemory`: Tests memory types and configurations.

**Key Test Functions:**
- `test_buffer_memory_initialization`: Tests ConversationBufferMemory
- `test_buffer_window_memory_initialization`: Tests ConversationBufferWindowMemory
- `test_memory_retention`: Tests conversation history retention

**Edge Cases:**
- Memory size limits
- Memory persistence between sessions

#### `test_agent_tools.py` and `test_agent_custom_tool_prompt.py`

**Purpose:** Tests Agent's ability to use tools and custom prompts for tools.

**Test Classes:**
- `TestAgentTools`: Tests tool registration and invocation
- `TestAgentCustomToolPrompt`: Tests custom prompts for tools

**Key Test Functions:**
- `test_tool_registration`: Tests adding tools to the agent
- `test_process_with_tools`: Tests using tools in responses
- `test_process_with_custom_prompt`: Tests tool use with custom prompts

**Edge Cases:**
- Tool failure handling
- Custom prompt formatting
- Multiple tool interaction

### Configuration Tests

#### `test_config_manager.py`

**Purpose:** Tests configuration management system.

**Test Classes:**
- `TestConfigManager`: Tests loading, updating, and validating configurations.

**Key Test Functions:**
- `test_initialization`: Tests ConfigManager initialization
- `test_get_config`: Tests retrieving configurations
- `test_update_config`: Tests updating configurations
- `test_validate_config`: Tests configuration validation

**Edge Cases:**
- Invalid configurations
- Missing configurations
- Deep update operations

### Workflow Tests

#### `test_workflow.py`

**Purpose:** Tests Workflow class for step management.

**Test Classes:**
- `TestWorkflow`: Tests workflow creation and management.

**Key Test Functions:**
- `test_initialization`: Tests Workflow initialization
- `test_add_step`: Tests adding steps to workflows
- `test_get_step`: Tests retrieving steps
- `test_link_steps`: Tests linking steps together

**Edge Cases:**
- Missing steps
- Cycle detection
- Step replacement

#### `test_workflow_execution.py`

**Purpose:** Tests executing workflows.

**Test Classes:**
- `TestWorkflowExecution`: Tests workflow execution functionality.

**Key Test Functions:**
- `test_execute_workflow`: Tests running complete workflows
- `test_workflow_with_parallel_steps`: Tests parallel step execution
- `test_workflow_with_conditional_links`: Tests conditional execution paths

**Edge Cases:**
- Error handling during execution
- Step execution order
- Resource cleanup after execution

### Step Tests

#### `test_step.py` and `test_create_step.py`

**Purpose:** Tests Step class functionality and creation.

**Test Classes:**
- `TestStep`: Tests Step base class functionality
- `TestCreateStep`: Tests step creation process

**Key Test Functions:**
- `test_name_attribute_exist`: Verifies Step name attribute
- `test_data_storage_command_line_name`: Tests DataStorageCommandLine name
- `test_link_direct_uses_name`: Tests LinkDirect access to name attributes

**Edge Cases:**
- Default name handling
- Name validation
- Output management

### Data Storage Tests

#### `test_data_storage_base.py` and `test_data_storage_command_line.py`

**Purpose:** Tests data storage mechanisms.

**Test Classes:**
- `TestDataStorageBase`: Tests DataStorageBase functionality
- `TestDataStorageCommandLine`: Tests command line I/O

**Key Test Functions:**
- `test_initialization`: Tests object initialization
- `test_get_set`: Tests data getting and setting
- `test_process`: Tests data processing

**Edge Cases:**
- Invalid data handling
- Data type conversions
- Special character handling

## Builder Tests

### `test_create_step_integration.py`

**Purpose:** Integration tests for CreateStep functionality.

**Test Classes:**
- `TestCreateStepIntegration`: Tests end-to-end step creation.

**Key Test Functions:**
- `test_create_step_execute`: Tests step creation including file generation with correct directory structure
- `non_interactive_start_monitoring`: Utility for non-interactive testing

**Edge Cases:**
- Non-interactive operation
- File creation verification
- Template customization
- Correct step naming and directory structure

### `test_create_step_execution.py`

**Purpose:** Tests execution details of CreateStep.

**Test Classes:**
- `TestCreateStepExecution`: Tests step creation execution behavior.

**Key Test Functions:**
- `test_create_step_execute_setup`: Tests setup of required objects
- `test_agent_builder_debug_mode_setting`: Tests debug mode propagation
- `test_connection_strength_debug_mode_setting`: Tests debug mode configuration

**Edge Cases:**
- Debug mode handling
- Proper mock setup for async operations

### `test_workflow_steps.py`

**Purpose:** Tests workflow step creation and management.

**Test Classes:**
- `TestCreateStep`: Tests workflow step creation scenarios.

**Key Test Functions:**
- `test_create_step_basic`: Tests basic step creation, ensuring the step class name is formatted correctly as "StepTestStep"
- `test_create_step_no_workflow`: Tests creation without workflow
- `test_create_step_existing_directory`: Tests creation with existing directory
- `test_create_step_with_description`: Tests creation with description, verifying the step class name and file paths
- `test_create_step_simplified`: Simplified test with direct patches

**Edge Cases:**
- Missing workflow handling
- Directory conflict resolution
- Input validation
- Correct step naming format (camel case without underscores)

### `test_agent_workflow_builder.py`

**Purpose:** Tests the AgentWorkflowBuilder class.

**Test Classes:**
- `TestAgentWorkflowBuilder`: Tests builder functionality.

**Key Test Functions:**
- `test_process`: Tests processing user input and returning responses
- `test_component_reuse_functionality`: Tests the system's ability to prioritize existing component reuse when appropriate
- `test_is_requesting_new_class`: Tests detection of requests to create new classes from scratch
- `test_should_generate_code`: Tests code generation decision logic
- `test_suggest_implementation`: Tests the implementation suggestion system with both existing and new components

**Edge Cases:**
- Pattern matching for detecting new class requests
- Component similarity assessment
- Code generation decision making logic
- Synchronous and asynchronous execution paths (with sync wrappers for async tests)

### `test_asyncio_issues.py`

**Purpose:** Tests and ensures proper async/await usage.

**Test Classes:**
- `TestAsyncioIssues`: Tests for asyncio-related issues.

**Key Test Functions:**
- `test_nested_event_loop_prevention`: Tests against nested event loops
- `test_create_workflow_is_async`: Ensures proper async implementation
- `test_create_workflow_implementation`: Tests async workflow creation with proper awaiting
- `test_all_builder_methods_are_properly_async`: Validates that all builder methods are properly implemented as async

**Edge Cases:**
- Nested event loop detection
- Async method signatures
- Coroutine execution
- Proper use of await vs asyncio.run()

## Integration Tests

### `test_workflow.py` (Integration)

**Purpose:** End-to-end tests of simple workflows.

**Test Functions:**
- `test_workflow`: Tests a complete workflow with two connected steps

**Edge Cases:**
- Data flow through multiple steps
- Error propagation
- Execution order

### `test_workflow_advanced.py`

**Purpose:** Tests more complex workflow scenarios.

**Test Functions:**
- `test_workflow_advanced`: Tests workflows with multiple steps and complex connections

**Edge Cases:**
- Multiple step interaction
- Parallel execution
- Conditional branching

### `test_step_creation.py` and Variants

**Purpose:** Tests step creation in different contexts.

**Test Functions:**
- `test_step_creation`: Tests basic step creation
- `test_step_creation_with_agent`: Tests creation using Agent
- `test_step_creation_with_custom_input`: Tests custom inputs
- `test_step_creation_with_complex_input`: Tests complex input handling

**Edge Cases:**
- User input simulation
- Invalid input handling
- Complex step requirements

### `test_component_reuse.py` (Integration)

**Purpose:** Tests component reuse in integrated contexts.

**Test Classes:**
- `TestComponentReuse`: Tests component discovery and reuse.

**Key Test Functions:**
- `test_list_existing_components`: Tests component discovery
- `test_agent_code_writer_prioritizes_existing`: Tests reuse prioritization
- `test_workflow_builder_suggests_existing`: Tests reuse suggestion

**Edge Cases:**
- Component similarity assessment
- Version compatibility
- Suggestion relevance

## Mock Components

### Mock Classes

The test suite uses several mock classes:

1. **MockStep**: Mock step class for testing links and triggers
2. **MockDataUnit**: Mock data unit for testing data flow
3. **MockExecutor**: Mock executor for testing step execution
4. **MockAgent**: Mock agent for testing without requiring LLM access
5. **MockNanoBrainBuilder**: Mock builder for simplified testing

### Mocking Strategies

1. **Function Mocking**:
   - `non_interactive_start_monitoring`: Creates non-interactive monitoring
   - `create_safe_async_mock`: Creates AsyncMocks that prevent warnings

2. **Class Mocking**:
   - Comprehensive mock implementations of core classes
   - Mock class substitution for external dependencies

3. **Patching**:
   - Extensive use of `unittest.mock.patch`
   - Monkey patching for method overrides

## Test Infrastructure

### Helper Functions

- `setup_llm_mocks()`: Sets up mock LLM to avoid API key requirements
- `create_safe_async_mock()`: Creates AsyncMock objects that don't produce warnings
- `async_test`: Decorator for running async tests with proper event loop management

### Test Environment Setup

- `unittest.IsolatedAsyncioTestCase` for async tests
- Temporary test directories
- Environment variable configuration (NANOBRAIN_TESTING='1')

## Areas for Improvement

### Test Coverage

1. **Error Conditions**:
   - Add more tests for error handling and recovery
   - Test boundary conditions more thoroughly
   - Test with invalid inputs

2. **Performance Testing**:
   - Add tests for performance under load
   - Test with large datasets
   - Test concurrent operations

3. **Security Testing**:
   - Test input validation and sanitization
   - Test permission handling
   - Test resource isolation

### Test Organization

1. **Structure**:
   - Group tests more systematically
   - Reduce test interdependencies
   - Improve test naming conventions

2. **Documentation**:
   - Add more detailed docstrings
   - Document test preconditions and postconditions
   - Add test case diagrams for complex scenarios

### Test Isolation

1. **Resource Management**:
   - Improve cleanup between tests
   - Better handling of async resource cleanup
   - Reduce test side effects

2. **Mock Enhancement**:
   - Create more comprehensive mock implementations
   - Better handling of complex async mock scenarios
   - Improved simulation of external dependencies

### Specific Component Improvements

1. **LinkBase and Trigger Tests**:
   - Add tests for concurrent trigger activations
   - Test more complex trigger conditions
   - Test with various reliability settings

2. **Workflow Tests**:
   - Test more complex workflow topologies
   - Test with very large workflows
   - Test workflow versioning

3. **Agent Tests**:
   - Test with more diverse prompt types
   - Test with different LLM providers
   - Test memory management more thoroughly

4. **Builder Tests**:
   - Test more complex build scenarios
   - Test with real-world templates
   - Test build optimization

5. **Integration Tests**:
   - Add more comprehensive end-to-end tests
   - Test integration with external systems
   - Test distributed operation

### Test Infrastructure Improvements

1. **Testing Framework**:
   - Consider adding pytest fixtures
   - Implement parameterized tests
   - Add support for test parallelization

2. **Mocking**:
   - Create a more systematic mocking framework
   - Improve async mock behavior
   - Better handling of complex dependency chains

3. **CI/CD Integration**:
   - Add test metrics collection
   - Implement test result visualization
   - Add coverage tracking

By addressing these areas for improvement, the Nanobrain test suite can become more comprehensive, reliable, and maintainable, ultimately leading to a more robust framework. 