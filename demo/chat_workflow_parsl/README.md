# Parsl Chat Workflow Demo

Interactive demonstration of the NanoBrain Parsl Chat Workflow, showcasing distributed conversational AI processing using Parsl.

## Overview

This demo provides a hands-on experience with the Parsl Chat Workflow, demonstrating:

- **Distributed Agent Processing**: Multiple conversational agents working in parallel
- **Parsl Integration**: Real-time distributed execution using Parsl framework
- **Performance Monitoring**: Live metrics and statistics
- **Interactive Interface**: Command-line interface with various features
- **Load Testing**: Batch processing capabilities for performance evaluation

## Quick Start

### Prerequisites

1. **NanoBrain Framework**: Ensure the NanoBrain framework is properly installed
2. **Parsl**: Install Parsl for distributed computing
3. **Python 3.8+**: Required for async/await support

### Installation

```bash
# From the project root directory
cd demo/chat_workflow_parsl

# Install dependencies (if not already installed)
pip install parsl pyyaml

# Run the demo
python run_parsl_chat_demo.py
```

## Demo Features

### Interactive Chat

Engage in natural conversation with the distributed agent system:

```
🗣️  You: What is machine learning?
⚡ Processing with Parsl distributed agents...
🤖 Response: [Aggregated response from multiple agents]
```

### Performance Commands

- **`/stats`**: View real-time performance statistics
- **`/status`**: Check workflow component status
- **`/batch N`**: Run load test with N messages

### Help and Information

- **`/help`**: Display comprehensive help information
- **`/quit`**: Gracefully exit the demo

## Demo Workflow

### 1. Initialization Phase

```
🚀 Initializing NanoBrain Parsl Chat Workflow
============================================================
✅ Workflow initialized: ParslChatWorkflow
   - Agents: 3
   - Parsl Executor: ✅
   - Data Units: 6
   - Parsl Apps: 2
```

### 2. Interactive Session

```
🎯 Parsl Chat Workflow Features:
  • Distributed execution with multiple agents
  • Parsl-based parallel processing
  • Performance monitoring and metrics
  • Conversation history with metadata
  • Load balancing across agents

📋 Available Commands:
  /help     - Show this help
  /stats    - Show performance statistics
  /status   - Show workflow status
  /batch N  - Send N test messages for load testing
  /quit     - Exit the demo
```

### 3. Message Processing

Each user message is processed through:
1. **Input Collection**: User input captured via CLI
2. **Load Balancing**: Request distributed across available agents
3. **Parallel Processing**: Multiple agents process simultaneously using Parsl
4. **Response Aggregation**: Results combined into coherent response
5. **History Storage**: Conversation stored with execution metadata

## Performance Monitoring

### Real-time Statistics

```
📊 Performance Statistics
------------------------------
Total Requests: 5
Avg Processing Time: 1.234s
Agent Count: 3
Successful Responses: 3
Parsl Execution: ✅
Last Update: 2024-06-10T13:15:30
```

### Workflow Status

```
🔍 Workflow Status
--------------------
Name: ParslChatWorkflow
Initialized: ✅
Running: ✅
Agent Count: 3
Parsl Executor: ✅
Data Units: user_input, parsl_agent_input, parsl_agent_output, cli_output, conversation_history, parsl_performance_metrics
Parsl Apps: chat_processing, response_aggregation
```

## Load Testing

### Batch Processing

Test the system's performance with multiple concurrent requests:

```bash
# Run batch test with 10 messages
/batch 10
```

Sample output:
```
🔄 Running batch test with 10 messages...
📝 [1/10] Processing: What is artificial intelligence?...
✅ [1/10] Response received (156 chars)
📝 [2/10] Processing: How does machine learning work?...
✅ [2/10] Response received (203 chars)
...

📊 Batch Test Results:
   Messages: 10
   Total Time: 12.45s
   Avg Time/Message: 1.25s
```

## Agent Specialization

The demo showcases three specialized agents:

### Agent 1 (Creative)
- **Temperature**: 0.8 (high creativity)
- **Focus**: Creative and innovative responses
- **Specialties**: Art, design, storytelling, brainstorming

### Agent 2 (Analytical)
- **Temperature**: 0.3 (high precision)
- **Focus**: Analytical and fact-based responses
- **Specialties**: Data analysis, technical explanations, logic

### Agent 3 (Balanced)
- **Temperature**: 0.7 (balanced approach)
- **Focus**: General-purpose helpful responses
- **Specialties**: General conversation, explanations, assistance

## Configuration

### Demo Configuration

The demo uses the main workflow configuration at:
```
library/workflows/chat_workflow_parsl/ParslChatWorkflow.yml
```

### Customization

You can customize the demo by:

1. **Modifying Agent Configurations**: Edit files in `config/` directory
2. **Adjusting Parsl Settings**: Update executor configuration
3. **Changing Test Messages**: Modify the batch test message list
4. **Adding New Commands**: Extend the command handler

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'parsl'
   ```
   **Solution**: Install Parsl with `pip install parsl`

2. **Configuration Errors**
   ```
   FileNotFoundError: ParslChatWorkflow.yml
   ```
   **Solution**: Ensure you're running from the correct directory

3. **Agent Initialization Failures**
   ```
   Failed to initialize workflow: Agent configuration error
   ```
   **Solution**: Check agent configuration files in `config/` directory

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export NANOBRAIN_LOG_LEVEL=DEBUG
python run_parsl_chat_demo.py
```

## Example Session

```bash
$ python run_parsl_chat_demo.py

🚀 NanoBrain Parsl Chat Workflow Demo
==================================================
Demonstrating distributed chat processing with Parsl
Using existing NanoBrain infrastructure and best practices

✅ Configuration manager loaded successfully
🚀 Initializing NanoBrain Parsl Chat Workflow
============================================================
✅ Workflow initialized: ParslChatWorkflow
   - Agents: 3
   - Parsl Executor: ✅
   - Data Units: 6
   - Parsl Apps: 2

🎯 Parsl Chat Workflow Features:
  • Distributed execution with multiple agents
  • Parsl-based parallel processing
  • Performance monitoring and metrics
  • Conversation history with metadata
  • Load balancing across agents

🗣️  You: How does distributed computing work?
⚡ Processing with Parsl distributed agents...
🤖 Response: [parsl_agent_2] Processing your message with analytical and precise approach: How does distributed computing work? (Processed using distributed execution with gpt-3.5-turbo)

🗣️  You: /stats
📊 Performance Statistics
------------------------------
Total Requests: 1
Avg Processing Time: 0.156s
Agent Count: 3
Successful Responses: 3
Parsl Execution: ✅
Last Update: 2024-06-10T13:20:15

🗣️  You: /quit

🔄 Shutting down workflow...
✅ Shutdown complete

👋 Goodbye!
```

## Next Steps

After exploring the demo:

1. **Examine the Code**: Review the workflow implementation in `library/workflows/chat_workflow_parsl/`
2. **Customize Agents**: Modify agent configurations for different behaviors
3. **Scale Up**: Try running on multiple nodes or with more agents
4. **Integrate**: Use the workflow in your own applications
5. **Contribute**: Add new features or improvements to the workflow

## Support

For questions or issues:

1. Check the main workflow README: `library/workflows/chat_workflow_parsl/README.md`
2. Review the NanoBrain documentation
3. Examine the Parsl documentation for distributed computing concepts
4. Look at the source code for implementation details

## Recent Fixes

### Parsl Workflow Logging Configuration Fix

**Issue**: Parsl workflow and debug information was being displayed in the CLI/console even when global configuration was set to `mode: "file"`.

**Root Cause**: The Parsl workflow was not respecting the global logging configuration and Parsl itself was outputting debug information to the console.

**Fix Applied**:
- Updated `ParslChatWorkflow.__init__()` to load and respect global logging configuration
- Added `_setup_logging()` method to configure logging based on global settings
- Added `_configure_parsl_logging()` method to suppress Parsl console output when in file-only mode
- Updated `ParslDistributedAgent` to respect global logging configuration
- Modified Parsl apps to suppress all console output in remote workers
- Updated agent configuration to disable debug/logging for remote workers based on global settings

**Files Modified**:
- `nanobrain/library/workflows/chat_workflow_parsl/workflow.py`

**Key Changes**:
1. **Global Configuration Loading**: Workflow now loads global config and checks `logging.mode` setting
2. **Parsl Logger Configuration**: Removes console handlers from Parsl loggers when `mode: "file"`
3. **Remote Worker Silence**: Parsl apps redirect stdout/stderr to suppress console output
4. **Agent Configuration**: Agents are configured with appropriate logging settings based on global config

**Verification**: Run `python test_parsl_logging.py` to verify the logging configuration works correctly.

### ConversationHistoryUnit Initialization Fix

**Issue**: `TypeError: ConversationHistoryUnit.__init__() got an unexpected keyword argument 'db_path'`

**Root Cause**: The `ConversationHistoryUnit` constructor expects a `config` parameter (dict) containing the configuration, not direct keyword arguments like `db_path`.

**Fix Applied**:
- Updated `nanobrain/library/workflows/chat_workflow/chat_workflow.py` line 129
- Changed from: `ConversationHistoryUnit(db_path="chat_workflow_history.db")`
- Changed to: `ConversationHistoryUnit(config={'db_path': 'chat_workflow_history.db'})`

**Files Fixed**:
- `nanobrain/library/workflows/chat_workflow/chat_workflow.py`
- `nanobrain/library/test_library_structure.py`

**Verification**: Run `python test_fix_simple.py` to verify the fix works correctly.

## Files in this Directory

### Test Files
- `test_fix_simple.py` - Simple test to verify ConversationHistoryUnit fix
- `test_parsl_logging.py` - Test to verify Parsl workflow logging configuration
- `test_workflow_with_config.py` - Comprehensive test of Parsl workflow with configuration
- `test_chat_workflow_import.py` - Test chat workflow imports
- `test_updated_chat_workflow.py` - Test updated chat workflow functionality
- `run_chat_workflow_demo.py` - Demo script for chat workflow

### Parsl-Specific Files
- `run_parsl_chat_demo.py` - Parsl chat workflow demo
- `run_parsl_chat_demo_fixed.py` - Fixed version of Parsl demo
- `test_parsl_agent_direct.py` - Direct test of Parsl agent functionality

### Logging and Debug Files
- `test_centralized_logging.py` - Test centralized logging configuration
- `test_logging_fix.py` - Test logging fixes
- `test_workflow_quiet.py` - Test workflow with quiet logging
- `debug_test.py` - Simple debug test

### Comprehensive Tests
- `run_comprehensive_demo.py` - Comprehensive workflow demonstration
- `test_refactored_workflow.py` - Test refactored workflow components
- `test_simple_workflow.py` - Simple workflow test
- `test_workflow_simple.py` - Another simple workflow test

### Directories
- `logs/` - Log files from workflow runs
- `runinfo/` - Parsl run information and metadata

## Usage

### Basic Chat Workflow Test
```bash
python test_fix_simple.py
```

### Test Parsl Logging Configuration
```bash
python test_parsl_logging.py
```

### Run Chat Workflow Demo
```bash
python run_chat_workflow_demo.py
```

### Run Parsl Chat Demo
```bash
python run_parsl_chat_demo.py
```

## Architecture

The chat workflow demonstrates proper NanoBrain architecture with:

1. **Modular Components**: Agents, data units, executors
2. **Configuration-Driven**: YAML-based configuration files
3. **Proper Initialization**: Correct parameter passing to constructors
4. **Error Handling**: Graceful handling of missing dependencies
5. **Logging Integration**: Centralized logging configuration

## Configuration

The workflow uses the global configuration from `config/global_config.yml` for:
- Logging settings (file vs console output)
- API key management
- Performance monitoring settings

## Dependencies

- NanoBrain framework (installed as package)
- Optional: Parsl for distributed processing
- Optional: OpenAI API key for actual LLM responses

## Troubleshooting

### ConversationHistoryUnit Errors
If you see `TypeError: ConversationHistoryUnit.__init__() got an unexpected keyword argument`, ensure you're using the correct initialization pattern:

```python
# Correct
history_unit = ConversationHistoryUnit(config={'db_path': 'path/to/db.db'})

# Incorrect
history_unit = ConversationHistoryUnit(db_path='path/to/db.db')
```

### Import Errors
Ensure the NanoBrain package is properly installed:
```bash
pip install -e .
```

### Logging Issues
Check `config/global_config.yml` for logging configuration. Set `mode: "file"` to suppress console output. 