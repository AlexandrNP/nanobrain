# SystemModulator

Manages global system state parameters that influence component behavior.

Biological analogy: Neuromodulator systems in the brain (dopamine, serotonin, etc.)
Justification: Like how neuromodulators can broadly influence neural processing 
across the brain, this class provides global parameters that can affect many 
components simultaneously to tune system behavior.

## Methods

### get_modulator

```python
def get_modulator(self, name: str) -> float
```

Get current level of a specific system modulator.

### set_modulator

```python
def set_modulator(self, name: str, value: float)
```

Set level of a specific system modulator.

### update_from_event

```python
def update_from_event(self, event_type: str, magnitude: float)
```

Update modulator levels based on system events.

Biological analogy: How environmental events trigger neuromodulator release.
Justification: Different types of events should trigger appropriate system-wide
responses, similar to how stress increases norepinephrine in biological systems.

### apply_regulation

```python
def apply_regulation(self)
```

Apply homeostatic regulation to all modulators.

Biological analogy: Homeostatic return to baseline conditions.
Justification: Systems need to return to balanced states after perturbations
to maintain long-term stability.

