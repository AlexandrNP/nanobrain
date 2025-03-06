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

Checks if all input sources have received data.

Biological analogy: Synaptic integration.
Justification: Like how neurons integrate inputs from multiple synapses
to determine if firing threshold is reached, this trigger checks if
enough inputs are present to initiate execution.

