# TriggerDataChanged

Trigger that activates when input data of a link changes.

Biological analogy: Change detector neuron.
Justification: Like how some neurons in sensory systems respond
specifically to changes in input rather than constant stimuli,
this trigger activates when data changes rather than at all times.

## Methods

### check_condition

```python
def check_condition(self, **kwargs) -> bool
```

Checks if the input data has changed since last check.

Biological analogy: Neural adaptation and change detection.
Justification: Like how certain neurons adapt to constant stimuli
but respond vigorously to changes, this trigger ignores constant 
data but responds to changes.

