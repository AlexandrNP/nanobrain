# ConfigManager

Handles configuration via YAML files.

Biological analogy: Epigenetic mechanisms that control gene expression.
Justification: Like how epigenetic modifications determine which genes are expressed
without changing the DNA sequence, configuration parameters determine component
behavior without changing the underlying code.

## Methods

### get_config

```python
def get_config(self, class_name: str) -> Dict
```

Looks for <class_name>.yml in the appropriate directory and returns parameter dictionary.
First checks in local 'config' directory relative to base_path, then falls back to 'default_configs'.
Each parameter is supplemented with the property 'type' that references the class name.

Biological analogy: Cellular response to environmental cues.
Justification: Like how cells read their environment to determine appropriate
protein expression, components read configuration files to determine behavior.

### update_config

```python
def update_config(self, updates: Dict, adaptability_threshold: float) -> bool
```

Updates configuration if adaptability threshold is met.

Biological analogy: Cellular plasticity - ability to change in response to stimuli.
Justification: Components with higher adaptability should be more responsive to
configuration changes, similar to how more plastic neural circuits adapt more readily.

### adaptability

```python
def adaptability(self, value: float)
```

Set the adaptability level of the component.

