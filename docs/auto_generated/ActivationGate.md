# ActivationGate

Controls activation based on threshold and recovery dynamics.

Biological analogy: Neural membrane potential and action potential threshold.
Justification: Like how neurons have activation thresholds and refractory
periods, this gate controls when components can activate based on signal
strength and recovery state.

## Methods

### receive_signal

```python
def receive_signal(self, signal_strength: float) -> bool
```

Receives a signal and determines if it crosses the activation threshold.

Biological analogy: Neural integration and firing.
Justification: Like how neurons integrate inputs and fire if threshold
is reached, this method integrates signal strength with current level
and determines if activation occurs.

### decay

```python
def decay(self, amount: float)
```

Decays the current activation level toward resting level.

Biological analogy: Leaky integration in neurons.
Justification: Like how neural membrane potential decays over time,
this method reduces the current level toward the resting level.

