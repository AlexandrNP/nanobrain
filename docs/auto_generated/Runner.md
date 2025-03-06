# Runner

Base class for objects that can be executed.

Biological analogy: Neuron with activation potential.
Justification: Like how neurons can be activated to process and transmit
signals, runnable components can be executed to process and transmit data.

## Description

Base class for executable components that can be activated and run
by an executor. Manages activation state and execution flow.


## Biological Analogy

Functions like a neuron with activation potential, integrating inputs
and firing when conditions are met, followed by a recovery period.


## Default Configuration

```yaml
activation_threshold: 1.0
adaptability: 0.0
executor: None
input_channels: []
output_channels: []
recovery_period: 0.0
resting_level: 0.0
running: false
state: INACTIVE
```

## Configuration Validation

### Required Parameters

- `executor`

### Optional Parameters

- `activation_threshold`
- `resting_level`
- `recovery_period`
- `adaptability`
- `input_channels`
- `output_channels`
- `state`
- `running`

### Parameter Constraints

#### `activation_threshold`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `resting_level`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `recovery_period`

- type: `float`
- min: `0.0`

#### `adaptability`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `state`

- type: `str`
- enum: `['INACTIVE', 'ACTIVE', 'RECOVERING', 'BLOCKED', 'ENHANCED', 'DEGRADED', 'CONFIGURING']`

#### `running`

- type: `bool`

## Usage Examples

### Example 1

### Example 2

### Example 3

## Methods

### check_runnable_config

```python
def check_runnable_config(self) -> bool
```

Checks that class name is in executor's runnable_types set.

Biological analogy: Receptor-ligand specificity checking.
Justification: Like how a neurotransmitter binds only to neurons with
the matching receptors, executors can only execute runnables of
compatible types.

### get_config

```python
def get_config(self, class_dir: str) -> dict
```

Get configuration using ConfigManager.

### update_config

```python
def update_config(self, updates: dict, adaptability_threshold: float) -> bool
```

Update configuration using ConfigManager.

### adaptability

```python
def adaptability(self, value: float)
```

Set the adaptability level.

### state

```python
def state(self, value)
```

Set the current state.

### running

```python
def running(self, value: bool)
```

Set whether the component is running.

