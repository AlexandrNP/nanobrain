# Router

Routes one input to multiple outputs with adaptive routing and resource management.

Biological analogy: Combined axonal terminal and routing functionality.
Justification: Like how axon terminals manage neurotransmitter resources and
adaptively route signals to multiple targets, this router manages transmission
resources and adaptively routes data to multiple outputs.

## Description

Routes data from one input to multiple outputs with adaptive routing strategies
and resource management. Implements circuit breaker pattern for fault tolerance.


## Biological Analogy

Functions like axon terminals that adaptively route signals to multiple targets
while managing neurotransmitter resources and synaptic plasticity.


## Default Configuration

```yaml
adaptability: 0.0
error_threshold: 0.3
executor: None
exploration_rate: 0.0
fanout_reliability: 1.0
input_source: None
learning_rate: 0.0
load_threshold: 1.0
output_sinks: []
recovery_time: 0.0
resource_threshold: 0.0
routing_strategy: adaptive
running: false
state: INACTIVE
```

## Configuration Validation

### Required Parameters

- `executor`
- `input_source`
- `output_sinks`

### Optional Parameters

- `routing_strategy`
- `fanout_reliability`
- `learning_rate`
- `exploration_rate`
- `load_threshold`
- `error_threshold`
- `recovery_time`
- `resource_threshold`
- `state`
- `running`
- `adaptability`

### Parameter Constraints

#### `routing_strategy`

- type: `str`
- enum: `['adaptive', 'broadcast', 'random', 'weighted']`

#### `fanout_reliability`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `learning_rate`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `exploration_rate`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `load_threshold`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `error_threshold`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `recovery_time`

- type: `float`
- min: `0.0`

#### `resource_threshold`

- type: `float`
- min: `0.0`
- max: `1.0`

#### `state`

- type: `str`
- enum: `['INACTIVE', 'ACTIVE', 'RECOVERING', 'BLOCKED', 'ENHANCED', 'DEGRADED', 'CONFIGURING']`

#### `running`

- type: `bool`

#### `adaptability`

- type: `float`
- min: `0.0`
- max: `1.0`

## Usage Examples

### Example 1

### Example 2

### Example 3

## Methods

### get_metrics

```python
def get_metrics(self) -> Dict[str, Any]
```

Returns current routing metrics.

### check_runnable_config

```python
def check_runnable_config(self) -> bool
```

Check if router is properly configured.

### get_config

```python
def get_config(self, class_dir: str) -> dict
```

Get configuration from runner.

### update_config

```python
def update_config(self, updates: dict, adaptability_threshold: float) -> bool
```

Update configuration through runner.

### adaptability

```python
def adaptability(self, value: float)
```

Set adaptability through runner.

### state

```python
def state(self, value)
```

Set state through runner.

### running

```python
def running(self, value: bool)
```

Set running state through runner.

