# IConfigurable

Interface for components that can be configured via external configuration.

Biological analogy: Epigenetic mechanisms that control gene expression.
Justification: Like how epigenetic modifications determine which genes are expressed
without changing the DNA sequence, configuration parameters determine component
behavior without changing the underlying code.

## Methods

### get_config

```python
def get_config(self, class_dir: str) -> Dict
```

Get configuration from specified directory or default location.

### update_config

```python
def update_config(self, updates: Dict, adaptability_threshold: float) -> bool
```

Update configuration if adaptability threshold is met.

### adaptability

```python
def adaptability(self, value: float)
```

Set the adaptability level of the component.

