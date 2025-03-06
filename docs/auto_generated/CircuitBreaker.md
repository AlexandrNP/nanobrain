# CircuitBreaker

Implements the Circuit Breaker pattern to prevent cascade failures.

Biological analogy: Neural inhibitory circuits.
Justification: Like how inhibitory neurons prevent runaway excitation in neural
circuits, the circuit breaker prevents cascading failures in the system by 
blocking activation when too many errors occur.

## Methods

### record_success

```python
def record_success(self)
```

Record a successful operation and reset failure count.

Biological analogy: Successful processing reinforces normal function.
Justification: Successful neural operations reinforce normal activation
patterns and reduce inhibitory tone.

### record_failure

```python
def record_failure(self)
```

Record a failure and potentially open the circuit.

Biological analogy: Recruitment of inhibitory neurons after abnormal activity.
Justification: Like how seizure activity recruits inhibitory mechanisms,
repeated failures trigger protective inhibition.

### can_execute

```python
def can_execute(self) -> bool
```

Determines if execution should be allowed.

Biological analogy: Gating of neural activity based on circuit state.
Justification: Like how inhibitory gating controls information flow in
neural circuits, the circuit breaker gates execution based on system state.

### get_inhibition_level

```python
def get_inhibition_level(self) -> float
```

Returns the current level of inhibition (0.0-1.0).

Biological analogy: Strength of inhibitory tone in a neural circuit.
Justification: Inhibitory strength in neural circuits varies continuously,
affecting the likelihood of circuit activation.

