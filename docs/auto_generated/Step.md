# Step

Base class for workflow steps.

Biological analogy: Functional neural circuit.
Justification: Like how functional neural circuits process specific
types of information and pass results to other circuits, steps process
specific operations and pass results to other steps.

## Methods

### register_input_source

```python
def register_input_source(self, link_id: str, data_unit: DataUnitBase)
```

Register a new input source data unit with a specific ID.

Biological analogy: Synaptic connection formation.
Justification: Like how neurons form new synaptic connections,
steps can register new input sources.

Args:
    link_id: The identifier for this input source
    data_unit: The data unit that will provide input

### register_output

```python
def register_output(self, data_unit: DataUnitBase)
```

Register an output data unit.

Biological analogy: Axon terminal formation.
Justification: Like how neurons form axon terminals to connect
with target neurons, steps can register output data units.

Args:
    data_unit: The data unit that will receive output

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

