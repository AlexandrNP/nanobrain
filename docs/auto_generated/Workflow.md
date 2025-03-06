# Workflow

Container for multiple steps forming a workflow.

Biological analogy: Functional brain network.
Justification: Like how functional brain networks involve multiple
interconnected cortical areas working together to accomplish complex
tasks, workflows involve multiple interconnected steps working
together to accomplish complex processing.

## Methods

### apply_modulator_effects

```python
def apply_modulator_effects(self)
```

Apply effects of system modulators to the workflow.

Biological analogy: Neuromodulator effects on brain networks.
Justification: Like how brain-wide neuromodulators affect network
properties (e.g., dopamine affects working memory function),
system modulators affect workflow properties.

### modulate_step

```python
def modulate_step(self, step: Step)
```

Apply system modulator effects to a specific step.

Biological analogy: Regional neuromodulation.
Justification: Like how neuromodulators can have specific effects
on particular brain regions, system modulators can have specific
effects on particular workflow steps.

### organize_hierarchy

```python
def organize_hierarchy(self)
```

Organize steps into a hierarchical processing order.

Biological analogy: Hierarchical organization of brain networks.
Justification: Like how the brain organizes processing in hierarchical
networks (e.g., visual processing hierarchy), workflows organize
steps in dependency-based hierarchies.

