# TriggerAllDataReceived

Trigger that activates when all input data sources have received data.

Biological analogy: Integration neuron.
Justification: Like how some neurons only fire when receiving input from
multiple sources (e.g., coincidence detectors), this trigger only activates
when all data sources have provided input.

## Methods

### check_condition

```python
def check_condition(self, **kwargs) -> bool
```

Checks if all input sources in the step have received data.

Biological analogy: Synaptic integration.
Justification: Like how neurons integrate inputs from multiple synapses
to determine if firing threshold is reached, this trigger checks if
enough inputs are present to initiate execution.

### update_input_sources

```python
def update_input_sources(self)
```

Update knowledge of the step's input sources.

Biological analogy: Synaptic remodeling.
Justification: Like how neurons can form new synaptic connections
and prune existing ones, this trigger can update its understanding
of the step's input sources.

### get_config

```python
def get_config(self, class_dir: str) -> dict
```

Get configuration for this class.

### update_config

```python
def update_config(self, updates: dict, adaptability_threshold: float) -> bool
```

Update configuration if adaptability threshold is met.

### sensitivity

```python
def sensitivity(self, value)
```

Set trigger sensitivity.

### adaptation_rate

```python
def adaptation_rate(self, value)
```

Set adaptation rate.

### activation_gate

```python
def activation_gate(self, value)
```

Set the activation gate.

Biological analogy: Adjusting activation threshold.
Justification: Like how neurons can adjust their firing thresholds,
triggers can adjust their activation thresholds.

