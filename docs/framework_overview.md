# NanoBrain Framework Overview

## Introduction

NanoBrain is a biologically-inspired framework for building adaptive, resilient systems. It draws inspiration from neural systems to create software components that can adapt, learn, and recover from failures.

## Core Principles

1. **Biological Analogies**: Components mirror biological systems for intuitive understanding
2. **Adaptive Behavior**: Connection strengths adapt based on usage patterns
3. **Resilience**: Built-in circuit breakers and recovery mechanisms
4. **Modularity**: Components can be combined in various ways

## Framework Architecture

### Core Components

- **Configuration Management**: ConfigManager, DirectoryTracer, ConfigLoader
- **Data Units**: DataUnitBase, DataUnitMemory, DataUnitFile
- **Links**: LinkBase, LinkDirect, LinkFile
- **Execution**: Runner, Router, ExecutorBase, ExecutorFunc, ExecutorParallel
- **Agents**: Agent, Step, Workflow

### Data Flow

```
[DataUnit] → [Link] → [Runner/Router] → [Executor] → [Output DataUnit]
```

### Control Flow

```
[Trigger] → [Runner] → [Router] → [Multiple Links] → [Multiple DataUnits]
```

## LLM Integration

The framework includes robust integration with various Language Model providers:

- **OpenAI**: GPT models (gpt-3.5-turbo, gpt-4, etc.)
- **Anthropic**: Claude models (claude-2, etc.)
- **Google**: Gemini models
- **Meta/Llama**: Llama models
- **Mistral**: Mistral models

The Agent class can work with both chat-based models (BaseChatModel) and completion-based models (BaseLLM).

## Tool Calling

The framework supports two approaches to tool calling:

1. **LangChain Tool Binding**: Wrapping Step classes as LangChain tools
2. **Custom Tool Prompts**: Using custom prompts for tool calling

For more details, see [Tool Calling Documentation](tool_calling.md).

## Getting Started

To get started with the NanoBrain framework:

1. Create instances of the components you need
2. Configure them using the provided configuration options
3. Connect them together to create your desired data and control flow
4. Run your system and observe its adaptive behavior

## Documentation

- [Auto-generated Class Documentation](auto_generated/index.md)
- [UML Diagrams](UML.md)
- [Tool Calling](tool_calling.md)

