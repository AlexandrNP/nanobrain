# TriggerBase

Base class for components that detect conditions for activation.

Biological analogy: Sensory neuron.
Justification: Like how sensory neurons detect specific environmental
conditions and convert them to neural signals, triggers detect specific
computational conditions and convert them to workflow activations.

## Methods

### runnable

```python
def runnable(self, value)
```

Set the runnable object.

### check_condition

```python
def check_condition(self, **kwargs) -> bool
```

Checks if the condition for triggering is met.

Biological analogy: Sensory transduction.
Justification: Like how sensory neurons convert specific environmental
stimuli into neural signals, this method converts specific computational
conditions into boolean signals.

### get_config

```python
def get_config(self, class_dir: str) -> dict
```

Get configuration for this class.

### update_config

```python
def update_config(self, updates: dict, adaptability_threshold: float) -> bool
```

Delegate to config manager.

