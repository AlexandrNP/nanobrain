# Agent

LLM-powered agent that processes inputs using language models.

Biological analogy: Higher-order cognitive processing area.
Justification: Like how prefrontal cortex integrates information from multiple
sources and uses past experiences to generate adaptive responses, this agent
integrates inputs with context memory to generate intelligent responses.

## Description

LLM-powered agent that processes inputs using language models

## Biological Analogy

Higher-order cognitive processing area (prefrontal cortex)

## Justification

Like how the prefrontal cortex integrates information from multiple sources and uses past experiences to generate adaptive responses, this agent integrates inputs with context memory to generate intelligent responses.


## Objectives

- Process inputs using language models with contextual awareness
- Maintain conversation history with both full and working memory
- Generate coherent and contextually relevant responses
- Adapt behavior based on dynamic prompt templates and configuration
- Share and learn from collective agent experiences through shared context

## Default Configuration

```yaml
context_sensitivity: 0.8
creativity: 0.5
executor: ExecutorFunc
memory_key: chat_history
memory_window_size: 5
model_class: null
model_name: gpt-3.5-turbo
prompt_file: null
prompt_template: null
prompt_variables: {}
response_coherence: 0.7
shared_context_key: null
tools_config_path: null
use_buffer_window_memory: true
use_shared_context: false
```

## Configuration Validation

### Required Parameters

- `executor`

### Optional Parameters

- `model_name`
- `model_class`
- `memory_window_size`
- `prompt_file`
- `prompt_template`
- `prompt_variables`
- `use_shared_context`
- `shared_context_key`
- `tools_config_path`
- `use_buffer_window_memory`
- `memory_key`

### Parameter Constraints

#### `memory_window_size`

- min: `1`
- max: `10`
- type: `int`

#### `context_sensitivity`

- min: `0.0`
- max: `1.0`
- type: `float`

#### `creativity`

- min: `0.0`
- max: `1.0`
- type: `float`

#### `response_coherence`

- min: `0.0`
- max: `1.0`
- type: `float`

#### `prompt_file`

- type: `str`

#### `prompt_template`

- type: `str`

#### `use_shared_context`

- type: `bool`

#### `shared_context_key`

- type: `str`
- nullable: `True`

#### `tools_config_path`

- type: `str`
- nullable: `True`

#### `use_buffer_window_memory`

- type: `bool`

#### `memory_key`

- type: `str`

## Usage Examples

### Basic

Basic agent with default settings

```yaml
executor: ExecutorFunc
memory_window_size: 5
model_name: gpt-3.5-turbo
prompt_variables:
  role_description: assist users with general tasks
  specific_instructions: Maintain a helpful and professional tone
use_shared_context: false
```

### Technical

Technical expert agent with shared context

```yaml
context_sensitivity: 1.0
creativity: 0.3
model_name: gpt-4
prompt_file: prompts/technical
prompt_template: EXPERT_TEMPLATE
prompt_variables:
  expertise_areas: '- Software architecture and design

    - Performance optimization

    - Security best practices

    '
  technical_context: Enterprise software development environment
response_coherence: 0.9
shared_context_key: technical_team
tools_config_path: default_configs/AgentTools.yml
use_shared_context: true
```

### Creative

Creative assistant agent with shared context

```yaml
context_sensitivity: 0.7
creativity: 0.9
model_name: gpt-4
prompt_file: prompts/creative
prompt_template: CREATIVE_TEMPLATE
prompt_variables:
  creative_context: Digital media and design projects
  creative_domains: '- User interface design

    - Content creation

    - Visual storytelling

    '
response_coherence: 0.8
shared_context_key: creative_team
use_shared_context: true
```

## Methods

### Constructor

```python
def __init__(self, executor: ExecutorBase, model_name: str, model_class: Optional[str], memory_window_size: int, prompt_file: str, prompt_template: str, prompt_variables: Optional[Dict], use_shared_context: bool, shared_context_key: Optional[str], tools: Optional[List[Step]], use_custom_tool_prompt: bool, tools_config_path: Optional[str], use_buffer_window_memory: bool, memory_key: str, **kwargs)
```

Initialize the agent with LLM configuration.

Biological analogy: Neural circuit formation.
Justification: Like how neural circuits form with specific connectivity
patterns based on genetic and environmental factors, the agent initializes
with specific configuration parameters.

Args:
    executor: ExecutorBase instance for running steps
    model_name: Name of the LLM model to use
    model_class: Optional class name for the LLM model
    memory_window_size: Number of recent conversations to keep in context
    prompt_file: Path to file containing prompt template
    prompt_template: String containing prompt template
    prompt_variables: Variables to fill in the prompt template
    use_shared_context: Whether to use shared context between agents
    shared_context_key: Key for shared context group
    tools: Optional list of Step objects to use as tools
    use_custom_tool_prompt: Whether to use a custom prompt for tool calling
    tools_config_path: Path to YAML file with tool configurations
    use_buffer_window_memory: Whether to use ConversationBufferWindowMemory (True) or ConversationBufferMemory (False)
    memory_key: The key to use for the memory in the prompt template
    **kwargs: Additional keyword arguments

### get_full_history

```python
def get_full_history(self) -> List[Dict]
```

Get the full conversation history.

Biological analogy: Long-term memory retrieval.
Justification: Like how the brain can retrieve complete episodic
memories, this method retrieves the full conversation history.

### get_context_history

```python
def get_context_history(self) -> str
```

Get recent conversation history formatted as context.

Biological analogy: Working memory access.
Justification: Like how the brain maintains recent information in
working memory for immediate use, this method retrieves recent
conversation history for context.

### clear_memories

```python
def clear_memories(self)
```

Clear all memories.

Biological analogy: Memory reset.
Justification: Like how certain brain processes can clear working
memory for new tasks, this method clears the agent's memory.

### save_to_shared_context

```python
def save_to_shared_context(self, context_key: str)
```

Save memory to shared context.

Biological analogy: Collective memory formation.
Justification: Like how social organisms contribute to collective
knowledge, this method saves memory to a shared context.

### load_from_shared_context

```python
def load_from_shared_context(self, context_key: str)
```

Load memory from shared context.

Biological analogy: Social learning.
Justification: Like how organisms can learn from shared knowledge,
this method loads memory from a shared context.

### add_tool

```python
def add_tool(self, step: Step)
```

Add a new tool to the agent.

Biological analogy: Tool acquisition.
Justification: Like how organisms can learn to use new tools over time,
this method adds a new tool to the agent's repertoire.

### remove_tool

```python
def remove_tool(self, step: Step)
```

Remove a tool from the agent.

Biological analogy: Tool disuse.
Justification: Like how organisms may stop using certain tools when they're
no longer needed, this method removes a tool from the agent's repertoire.

### get_shared_context

```python
def get_shared_context(cls, context_key: str) -> List[Dict]
```

Get shared context by key.

Biological analogy: Accessing collective memory.
Justification: Like how social organisms access collective knowledge,
this method retrieves shared memory by key.

### clear_shared_context

```python
def clear_shared_context(cls, context_key: Optional[str])
```

Clear shared context.

Biological analogy: Collective memory reset.
Justification: Like how social groups can reset collective understanding,
this method clears shared memory.

### update_workflow_context

```python
def update_workflow_context(self, workflow_path: str)
```

Update the agent's context with information about the current workflow.

Biological analogy: Contextual awareness in the prefrontal cortex.
Justification: Like how the prefrontal cortex maintains awareness of the
current task context, this method updates the agent's context with
information about the current workflow.

Args:
    workflow_path: Path to the current workflow file

