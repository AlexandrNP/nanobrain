# TriggerStart

Trigger that always returns True, starting execution immediately.

Biological analogy: Pacemaker neuron.
Justification: Like how pacemaker neurons spontaneously initiate signals
without external input (e.g., in the cardiac system), this trigger
spontaneously initiates workflow execution.

## Methods

### check_condition

```python
def check_condition(self, **kwargs) -> bool
```

Almost always returns True, with some randomness to model 
biological variability.

Biological analogy: Spontaneous firing in pacemaker neurons.
Justification: Pacemaker neurons have intrinsic activity with
some biological variability in their firing patterns.

