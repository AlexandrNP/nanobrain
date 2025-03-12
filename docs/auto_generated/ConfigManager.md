# ConfigManager

Handles configuration via YAML files and creates class instances.

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

### create_instance

```python
def create_instance(self, class_name: str, **kwargs) -> Any
```

Factory method that creates an instance of the specified class using configuration.
First looks for a config file, then loads the class from a .py file with the same name,
and finally creates an instance with the config parameters.

Biological analogy: Protein synthesis from genetic instructions.
Justification: Like how cells synthesize proteins based on DNA templates modified
by epigenetic factors, this method creates objects based on class definitions
modified by configuration parameters.

Args:
    class_name: Name of the class to instantiate
    **kwargs: Additional parameters to override config values
    
Returns:
    An instance of the specified class
    
Raises:
    ImportError: If the class module cannot be found
    AttributeError: If the class cannot be found in the module
    TypeError: If the class cannot be instantiated with the given parameters

### adaptability

```python
def adaptability(self, value: float)
```

Set the adaptability level of the component.

