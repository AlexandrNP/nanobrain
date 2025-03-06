# DataStorageBase

Base class for data storage operations that respond to triggers.

Biological analogy: Memory system with retrieval mechanism.
Justification: Like how memory systems in the brain store information and
retrieve it in response to specific cues, this class stores data and
produces output in response to trigger conditions.

## Description

Base class for data storage operations that respond to triggers

## Biological Analogy

Memory system with retrieval mechanism

## Justification

Like how memory systems in the brain store information and retrieve it in response to specific cues, this class stores data and produces output in response to trigger conditions.


## Objectives

- Store and retrieve data based on trigger conditions
- Maintain a history of query-response interactions
- Provide a flexible base for different types of data storage mechanisms
- Connect input and output data units through a processing pipeline

## Default Configuration

```yaml
auto_monitor: true
max_history_size: 10
persistence_level: 0.7
retrieval_speed: 0.8
storage_capacity: 1.0
```

## Configuration Validation

### Required Parameters

- `executor`
- `input_unit`
- `output_unit`
- `trigger`

### Optional Parameters

- `max_history_size`
- `persistence_level`
- `retrieval_speed`
- `storage_capacity`
- `auto_monitor`

### Parameter Constraints

#### `max_history_size`

- min: `1`
- max: `100`
- type: `int`

#### `persistence_level`

- min: `0.0`
- max: `1.0`
- type: `float`

#### `retrieval_speed`

- min: `0.0`
- max: `1.0`
- type: `float`

#### `storage_capacity`

- min: `0.0`
- max: `1.0`
- type: `float`

#### `auto_monitor`

- type: `bool`

## Usage Examples

### Basic

Basic data storage with default settings

```yaml
auto_monitor: true
max_history_size: 10
persistence_level: 0.7
retrieval_speed: 0.8
storage_capacity: 1.0
```

### High_capacity

High-capacity data storage with increased history size

```yaml
auto_monitor: true
max_history_size: 50
persistence_level: 0.9
retrieval_speed: 0.6
storage_capacity: 1.0
```

### Fast_retrieval

Fast-retrieval data storage optimized for speed

```yaml
auto_monitor: true
max_history_size: 5
persistence_level: 0.5
retrieval_speed: 1.0
storage_capacity: 0.7
```

## Methods

### Constructor

```python
def __init__(self, executor: ExecutorBase, input_unit: DataUnitBase, output_unit: DataUnitBase, trigger: TriggerBase, **kwargs)
```

Initialize the DataStorageBase.

Args:
    executor: The executor responsible for running this step
    input_unit: The data unit to read input from
    output_unit: The data unit to write output to
    trigger: The trigger that activates this storage operation
    **kwargs: Additional keyword arguments

### get_history

```python
def get_history(self) -> List[Dict]
```

Get the processing history.

Returns:
    List of processing history entries

### clear_history

```python
def clear_history(self)
```

Clear the processing history.

Biological analogy: Memory clearance.
Justification: Like how certain memory systems can be cleared or reset,
this method clears the processing history.

### get_last_interaction

```python
def get_last_interaction(self) -> Dict
```

Get the last query-response interaction.

Returns:
    Dictionary containing the last query and response

