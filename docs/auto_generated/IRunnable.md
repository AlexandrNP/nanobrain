# IRunnable

Interface for components that can be executed.

Biological analogy: Neuron with activation potential.
Justification: Like how neurons can be activated to process and transmit
signals, runnable components can be executed to process and transmit data.

## Methods

### check_runnable_config

```python
def check_runnable_config(self) -> bool
```

Check if the component is properly configured to run.

### state

```python
def state(self, value)
```

Set the current state of the runnable component.

### running

```python
def running(self, value: bool)
```

Set whether the component is currently running.

