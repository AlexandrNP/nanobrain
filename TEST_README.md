# NanoBrain Framework Test Suite Documentation

## Overview

This document provides comprehensive documentation for the NanoBrain framework test suite, including testing scope, guidelines, performance benchmarks, and best practices for maintaining and extending the test coverage.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Testing Scope](#testing-scope)
4. [Running Tests](#running-tests)
5. [Test Categories](#test-categories)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Testing Guidelines](#testing-guidelines)
8. [Continuous Integration](#continuous-integration)
9. [Adding New Tests](#adding-new-tests)
10. [Troubleshooting](#troubleshooting)

## Testing Philosophy

The NanoBrain framework follows a comprehensive testing approach that ensures:

- **Reliability**: All core components are thoroughly tested
- **Performance**: Performance characteristics are monitored and benchmarked
- **Integration**: Components work correctly together
- **Regression Prevention**: Changes don't break existing functionality
- **Documentation**: Tests serve as living documentation

## Test Structure

```
nanobrain/
├── src/
│   ├── agents/
│   │   ├── config/                     # Agent-specific YAML configurations
│   │   │   ├── step_coder.yml          # CodeWriterAgent configuration
│   │   │   ├── step_file_writer.yml    # FileWriterAgent configuration
│   │   │   ├── step_coder_enhanced.yml # Enhanced CodeWriterAgent config
│   │   │   └── step_file_writer_enhanced.yml # Enhanced FileWriterAgent config
│   │   ├── code_writer.py              # CodeWriterAgent implementation
│   │   ├── file_writer.py              # FileWriterAgent implementation
│   │   └── __init__.py
│   ├── config/
│   │   ├── templates/                  # General workflow templates
│   │   │   └── workflow_example.yml    # Complete workflow example
│   │   ├── component_factory.py       # YAML-based component creation
│   │   ├── schema_validator.py         # Configuration validation
│   │   └── yaml_config.py              # YAML configuration system
│   └── core/                           # Core framework components
├── tests/
│   ├── test_logging_system.py          # Logging and monitoring tests
│   ├── test_component_factory.py       # YAML component factory tests
│   ├── test_core_components.py         # Core framework component tests
│   ├── test_agents.py                  # Agent system tests
│   ├── test_steps.py                   # Step system tests
│   ├── test_executors.py               # Executor backend tests
│   ├── test_integration.py             # Integration tests
│   ├── test_performance.py             # Performance benchmarks
│   └── fixtures/                       # Test fixtures and data
├── demo/                               # Demonstration scripts (also serve as integration tests)
│   ├── logging_showcase.py             # Comprehensive logging demonstration
│   ├── yaml_factory_demo.py            # YAML component factory demonstration
│   ├── code_writer_yaml_demo.py        # CodeWriterAgent YAML configuration demo
│   └── code_writer_advanced.py         # Advanced agent interaction demo
└── TEST_README.md                      # This file
```

## Directory Structure for YAML Configurations

The NanoBrain framework follows a structured approach to YAML configuration files:

### Agent-Specific Configurations
- **Location**: `nanobrain/src/agents/config/`
- **Purpose**: Configuration files specific to individual agent implementations
- **Naming Convention**: `step_<agent_name>.yml` for basic configurations, `step_<agent_name>_enhanced.yml` for advanced configurations
- **Examples**:
  - `step_coder.yml` - Basic CodeWriterAgent configuration
  - `step_file_writer.yml` - Basic FileWriterAgent configuration
  - `step_coder_enhanced.yml` - Advanced CodeWriterAgent with comprehensive prompts
  - `step_file_writer_enhanced.yml` - Advanced FileWriterAgent with detailed workflows

### General Templates
- **Location**: `nanobrain/src/config/templates/`
- **Purpose**: Workflow templates and general configuration examples
- **Examples**:
  - `workflow_example.yml` - Complete workflow with multiple components

### Configuration Search Priority
The component factory searches for YAML files in the following order:
1. `src/agents/config/` - Agent-specific configurations (highest priority)
2. `agents/config/` - Agent-specific configurations (relative path)
3. `src/config/templates/` - General templates
4. `config/templates/` - General templates (relative path)
5. `templates/` - Fallback template directory
6. `.` - Current directory

This structure ensures that agent-specific configurations are found first, allowing for proper organization and avoiding conflicts between different types of configurations.

## Testing Scope

### Core Components Tested

#### 1. Logging System (`test_logging_system.py`)
- **Structured Logging**: JSON-formatted logs with timestamps and metadata
- **Execution Context Tracking**: Unique request IDs and nested operation tracking
- **Performance Metrics**: Automatic collection of timing and resource metrics
- **Agent Conversation Logging**: Complete tracking of agent interactions
- **Tool Call Logging**: Detailed logging of tool executions
- **Error Handling**: Comprehensive error context and debugging information
- **Async Support**: Full async/await pattern support

#### 2. Agent System (`test_agents.py`)
- **Agent Initialization**: Proper setup and configuration
- **LLM Integration**: OpenAI and other LLM provider integration
- **Tool Registration**: Function and agent tool registration
- **Conversation Management**: History tracking and context management
- **Performance Tracking**: Token usage, execution counts, error rates
- **Error Handling**: Graceful failure and recovery

#### 3. Step System (`test_steps.py`)
- **Data Processing**: Input/output data unit management
- **Trigger Activation**: Event-driven execution
- **Link Propagation**: Data flow between steps
- **Executor Integration**: Different execution backends
- **Performance Monitoring**: Processing time and throughput

#### 4. Executor Backends (`test_executors.py`)
- **Local Executor**: Async task execution
- **Thread Executor**: CPU-bound task execution
- **Process Executor**: Isolated process execution
- **Parsl Executor**: HPC and distributed execution
- **Resource Management**: Proper initialization and cleanup

#### 5. Integration Tests (`test_integration.py`)
- **Agent-Step Workflows**: Mixed agent and step processing
- **Multi-Agent Collaboration**: Agent-to-agent communication
- **Complex Data Flows**: Multi-step data processing pipelines
- **Error Propagation**: Error handling across components
- **Performance Under Load**: Stress testing and resource usage

### Test Coverage Goals

- **Unit Tests**: 90%+ coverage for core components
- **Integration Tests**: All major workflow patterns
- **Performance Tests**: Baseline performance characteristics
- **Error Handling**: All error paths and edge cases

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Install optional dependencies for full testing
pip install openai parsl  # For LLM and HPC testing
```

### Basic Test Execution

```bash
# Run all tests
cd nanobrain
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_logging_system.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run async tests specifically
python -m pytest tests/ -k "async" -v
```

### Advanced Test Options

```bash
# Run performance benchmarks
python -m pytest tests/test_performance.py -v --benchmark-only

# Run integration tests
python -m pytest tests/test_integration.py -v

# Run with detailed logging
python -m pytest tests/ -v -s --log-cli-level=DEBUG

# Run specific test categories
python -m pytest tests/ -m "unit" -v        # Unit tests only
python -m pytest tests/ -m "integration" -v # Integration tests only
python -m pytest tests/ -m "performance" -v # Performance tests only
```

## Test Categories

### Unit Tests
- **Scope**: Individual component functionality
- **Duration**: < 1 second per test
- **Dependencies**: Minimal external dependencies
- **Examples**: Logger initialization, agent configuration, step processing

### Integration Tests
- **Scope**: Component interaction and workflows
- **Duration**: 1-10 seconds per test
- **Dependencies**: Multiple components, may require external services
- **Examples**: Agent-step workflows, multi-agent collaboration

### Performance Tests
- **Scope**: Performance characteristics and benchmarks
- **Duration**: 10-60 seconds per test
- **Dependencies**: May require specific hardware or configurations
- **Examples**: Throughput testing, memory usage, execution time benchmarks

### End-to-End Tests
- **Scope**: Complete workflow scenarios
- **Duration**: 30-300 seconds per test
- **Dependencies**: Full system setup, external services
- **Examples**: Complete AI workflows, HPC job execution

### YAML Configuration Tests
- **Scope**: Component creation and configuration from YAML files
- **Duration**: < 5 seconds per test
- **Dependencies**: YAML configuration files, component factory system
- **Examples**: Agent configuration loading, step creation from templates, workflow assembly

#### CodeWriterAgent YAML Configuration Testing

The framework includes comprehensive tests for YAML-based configuration of CodeWriterAgent:

```bash
# Run CodeWriterAgent YAML configuration tests
python -m pytest tests/test_component_factory.py::TestComponentFactory::test_code_writer_yaml_config_loading -v
python -m pytest tests/test_component_factory.py::TestComponentFactory::test_code_writer_default_prompt_fallback -v
python -m pytest tests/test_component_factory.py::TestComponentFactory::test_step_coder_yaml_template -v

# Run demo showing YAML configuration loading
python demo/code_writer_yaml_demo.py
```

**Test Coverage:**
- ✅ Custom system prompts loaded from YAML
- ✅ Model parameters (temperature, max_tokens) configured via YAML
- ✅ Default prompt fallback when YAML doesn't specify system_prompt
- ✅ step_coder.yml template creates proper CodeWriterAgent instances (located in `agents/config/`)
- ✅ Component factory properly handles CodeWriterAgent class registration
- ✅ Configuration validation and error handling

**YAML Configuration Structure:**
Agent-specific YAML configurations are located in `nanobrain/src/agents/config/` directory, following the principle that configuration files should be in the same directory as the corresponding agent implementation.

**Example YAML Configuration:**
```yaml
# Located in: nanobrain/src/agents/config/step_coder.yml
name: "StepCoder"
class: "CodeWriterAgent"
config:
  name: "StepCoder"
  description: "Specialized agent for generating software code"
  model: "gpt-4-turbo"
  temperature: 0.2
  max_tokens: 4000
  system_prompt: |
    You are a specialized code generation agent for the NanoBrain framework.
    
    Your responsibilities:
    1. Generate high-quality, well-documented code
    2. Follow best practices and coding standards
    3. Provide clear explanations of generated code
```

## Performance Benchmarks

### Baseline Performance Metrics

#### Logging System
- **Log Entry Creation**: < 1ms per entry
- **Context Tracking**: < 0.1ms overhead per operation
- **Performance Metrics Collection**: < 0.5ms per metric
- **Memory Usage**: < 10MB for 10,000 log entries

#### Agent System
- **Agent Initialization**: < 500ms
- **Simple Processing**: < 100ms (without LLM calls)
- **Tool Call Execution**: < 50ms per tool call
- **Memory Usage**: < 50MB per agent instance

#### Step System
- **Step Initialization**: < 100ms
- **Data Processing**: < 10ms per operation
- **Trigger Activation**: < 5ms
- **Memory Usage**: < 20MB per step instance

#### Executor Performance
- **Local Executor**: 1000+ ops/sec
- **Thread Executor**: 100+ ops/sec (CPU-bound tasks)
- **Process Executor**: 50+ ops/sec (isolated tasks)
- **Parsl Executor**: Depends on cluster configuration

### Performance Test Examples

```python
# Example performance test
@pytest.mark.performance
async def test_agent_processing_performance():
    """Test agent processing performance under load."""
    agent = SimpleAgent(AgentConfig(name="perf_test"))
    await agent.initialize()
    
    start_time = time.time()
    tasks = [agent.process(f"Task {i}") for i in range(100)]
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    # Performance assertions
    assert duration < 10.0  # Should complete in under 10 seconds
    assert len(results) == 100
    assert all(result for result in results)
```

## Testing Guidelines

### Writing Effective Tests

1. **Test Naming**: Use descriptive names that explain what is being tested
   ```python
   def test_agent_initialization_with_valid_config():
   def test_step_execution_with_missing_input_data():
   def test_logging_system_handles_concurrent_operations():
   ```

2. **Test Structure**: Follow the Arrange-Act-Assert pattern
   ```python
   async def test_agent_tool_registration():
       # Arrange
       config = AgentConfig(name="test_agent")
       agent = SimpleAgent(config)
       
       # Act
       agent.register_function_tool(lambda x: x * 2, "double", "Double a number")
       
       # Assert
       assert "double" in agent.available_tools
   ```

3. **Async Testing**: Use proper async test patterns
   ```python
   @pytest.mark.asyncio
   async def test_async_operation():
       result = await some_async_function()
       assert result is not None
   ```

4. **Error Testing**: Test both success and failure cases
   ```python
   async def test_agent_handles_invalid_tool_call():
       agent = SimpleAgent(AgentConfig(name="test"))
       await agent.initialize()
       
       with pytest.raises(ValueError):
           await agent.execute("Use non_existent_tool")
   ```

### Test Data Management

1. **Use Fixtures**: Create reusable test data and configurations
   ```python
   @pytest.fixture
   async def initialized_agent():
       config = AgentConfig(name="test_agent")
       agent = SimpleAgent(config)
       await agent.initialize()
       yield agent
       await agent.shutdown()
   ```

2. **Temporary Resources**: Clean up temporary files and resources
   ```python
   @pytest.fixture
   def temp_log_file():
       with tempfile.NamedTemporaryFile(delete=False) as f:
           yield Path(f.name)
       Path(f.name).unlink(missing_ok=True)
   ```

### Mocking and Isolation

1. **Mock External Dependencies**: Use mocks for external services
   ```python
   @patch('openai.AsyncOpenAI')
   async def test_agent_without_real_llm(mock_openai):
       mock_client = AsyncMock()
       mock_openai.return_value = mock_client
       # Test agent behavior with mocked LLM
   ```

2. **Isolate Tests**: Ensure tests don't affect each other
   ```python
   def setup_method(self):
       """Reset global state before each test."""
       self.logger = NanoBrainLogger("test", debug_mode=True)
   
   def teardown_method(self):
       """Clean up after each test."""
       self.logger.clear_logs()
   ```

## Continuous Integration

### GitHub Actions Configuration

The framework uses GitHub Actions for continuous integration:

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        python -m pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Quality Gates

- **Test Coverage**: Minimum 85% coverage required
- **Performance**: No regression in baseline performance
- **Code Quality**: All tests must pass
- **Documentation**: New features require test documentation

## Adding New Tests

### For New Components

1. **Create Test File**: Follow naming convention `test_<component>.py`
2. **Add Test Categories**: Include unit, integration, and performance tests
3. **Update Documentation**: Add component to this README
4. **Add Fixtures**: Create reusable test fixtures

### For Bug Fixes

1. **Reproduce Bug**: Create a test that reproduces the bug
2. **Fix Implementation**: Implement the fix
3. **Verify Fix**: Ensure the test passes
4. **Add Regression Test**: Prevent future regressions

### For Performance Improvements

1. **Baseline Measurement**: Establish current performance
2. **Implement Improvement**: Make the performance changes
3. **Benchmark**: Measure improved performance
4. **Update Benchmarks**: Update performance expectations

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Issue: ModuleNotFoundError
# Solution: Ensure PYTHONPATH includes src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Async Test Issues
```bash
# Issue: RuntimeError: There is no current event loop
# Solution: Use pytest-asyncio plugin
pip install pytest-asyncio
```

#### Performance Test Failures
```bash
# Issue: Performance tests failing on slower systems
# Solution: Adjust performance thresholds or skip on CI
@pytest.mark.skipif(os.getenv('CI'), reason="Skip performance tests on CI")
```

#### Resource Cleanup Issues
```bash
# Issue: Tests leaving resources open
# Solution: Use proper fixtures and cleanup
@pytest.fixture
async def resource():
    r = create_resource()
    yield r
    await r.cleanup()
```

### Debugging Tests

1. **Verbose Output**: Use `-v` flag for detailed test output
2. **Logging**: Use `-s --log-cli-level=DEBUG` for detailed logs
3. **Specific Tests**: Run specific tests with `-k "test_name"`
4. **Debugging**: Use `--pdb` to drop into debugger on failures

### Getting Help

- **Documentation**: Check component documentation in `src/`
- **Examples**: Review demo scripts in `demo/`
- **Issues**: Create GitHub issues for test-related problems
- **Discussions**: Use GitHub discussions for testing questions

## Conclusion

This test suite ensures the NanoBrain framework maintains high quality, performance, and reliability. By following these guidelines and maintaining comprehensive test coverage, we can confidently develop and deploy AI workflows with the framework.

For questions or contributions to the test suite, please refer to the main project documentation and contribution guidelines. 