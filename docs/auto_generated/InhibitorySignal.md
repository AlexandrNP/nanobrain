# InhibitorySignal

Represents a blocking signal that prevents component activation.

Biological analogy: Inhibitory neurotransmitters (GABA, glycine).
Justification: Like how inhibitory neurotransmitters prevent neurons from
firing despite excitatory inputs, inhibitory signals block components
from activating despite receiving activation signals.

## Methods

### activate

```python
def activate(self)
```

Activate the inhibitory signal.

Biological analogy: Release of inhibitory neurotransmitters.
Justification: When activated, inhibitory neurons release transmitters
that temporarily prevent target neurons from firing.

### is_active

```python
def is_active(self) -> bool
```

Check if the inhibitory signal is still active.

Biological analogy: Duration of inhibitory postsynaptic potentials.
Justification: Inhibitory signals have a specific duration before dissipating.

### get_strength

```python
def get_strength(self) -> float
```

Get the current inhibitory strength.

Biological analogy: Decreasing concentration of inhibitory neurotransmitters.
Justification: The effect of inhibition gradually decreases as neurotransmitters
are cleared from the synaptic cleft.

