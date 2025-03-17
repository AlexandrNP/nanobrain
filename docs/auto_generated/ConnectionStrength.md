# ConnectionStrength

Models the strength of connections between components.

Biological analogy: Synaptic weights in neural networks.
Justification: Like how synaptic weights determine the strength of
connections between neurons, this class models the strength of
connections between components.

## Methods

### get_value

```python
def get_value(self) -> float
```

Get the current connection strength value.

Biological analogy: Synaptic efficiency measurement.
Justification: Like how the efficacy of a synapse can be measured,
this method provides the current strength of the connection.

Returns:
    The current connection strength as a float between min_strength and max_strength

### value

```python
def value(self) -> float
```

Property that returns the current connection strength.

This provides an alternative way to access the strength value.

Returns:
    The current connection strength as a float

### strengthen

```python
def strengthen(self, adaptability: float) -> float
```

Strengthen the connection based on successful activity.

Biological analogy: Long-Term Potentiation (LTP).
Justification: Like how successful synaptic transmission strengthens
neural connections, this method increases connection strength.

Args:
    adaptability: Rate of adaptation (0.0-1.0)
    
Returns:
    The updated connection strength

### weaken

```python
def weaken(self, adaptability: float) -> float
```

Weaken the connection based on failed activity.

Biological analogy: Long-Term Depression (LTD).
Justification: Like how failed synaptic transmission weakens
neural connections, this method decreases connection strength.

Args:
    adaptability: Rate of adaptation (0.0-1.0)
    
Returns:
    The updated connection strength

### increase

```python
def increase(self, amount: float) -> float
```

Increase connection strength.

Biological analogy: Long-Term Potentiation (LTP) in neural systems.
Justification: In the brain, frequently used connections strengthen over time,
improving efficiency for common operations.

### decrease

```python
def decrease(self, amount: float) -> float
```

Decrease connection strength.

Biological analogy: Long-Term Depression (LTD) in neural systems.
Justification: In the brain, rarely used connections weaken over time,
allowing resources to be reallocated to more important pathways.

### adapt

```python
def adapt(self, source_activity: float, target_activity: float, adaptation_rate: float) -> float
```

Adapt connection strength based on correlated activity.

Biological analogy: Hebbian learning - "Neurons that fire together, wire together"
Justification: This implements an adaptive learning mechanism similar to how
neural connections strengthen when input and output neurons activate together.

