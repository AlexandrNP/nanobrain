# Step

Base class for workflow steps.

Biological analogy: Functional neural circuit.
Justification: Like how functional neural circuits process specific
types of information and pass results to other circuits, steps process
specific operations and pass results to other steps.

## Methods

### get_result

```python
def get_result(self) -> Any
```

Get the most recent result.

Biological analogy: Neural circuit output.
Justification: Like how neural circuits maintain their output
state until new processing occurs, steps maintain their result
until new execution occurs.

### check_runnable_config

```python
def check_runnable_config(self) -> bool
```

Check if the step is properly configured to run.

Biological analogy: Neural circuit readiness check.
Justification: Like how neural circuits must have proper connections
to function, steps must have proper configuration to execute.

### adaptability

```python
def adaptability(self, value: float)
```

Set the adaptability level of the step.

Biological analogy: Modulation of neural plasticity.
Justification: Like how neuromodulators can adjust plasticity levels,
this setter adjusts the adaptability of the step.

### state

```python
def state(self, value)
```

Set the current state of the step.

Biological analogy: Neural circuit state transition.
Justification: Like how neural circuits transition between states,
steps transition between operational states.

### running

```python
def running(self, value: bool)
```

Set whether the step is currently running.

Biological analogy: Neural circuit activation control.
Justification: Like how neural circuits can be activated or deactivated,
steps can be set to running or not running.

