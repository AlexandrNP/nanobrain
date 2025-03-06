# ConnectionStrength

Models the strength of connections between components.

Biological analogy: Synaptic weights in neural networks.
Justification: Like how synaptic weights determine the strength of
connections between neurons, this class models the strength of
connections between components.

## Methods

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

