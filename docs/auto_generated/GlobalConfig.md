# GlobalConfig

Singleton class for managing global configuration settings.

Biological analogy: Endocrine system.
Justification: Like how the endocrine system regulates global body functions through
hormones that affect multiple systems, the GlobalConfig manages framework-wide
settings that affect multiple components.

## Methods

### load_config

```python
def load_config(self, config_path: Optional[str]) -> bool
```

Load configuration from a YAML file.

Args:
    config_path: Path to the configuration file. If None, uses default locations.
    
Returns:
    bool: True if configuration was loaded successfully, False otherwise.

### save_config

```python
def save_config(self, config_path: Optional[str]) -> bool
```

Save the current configuration to a YAML file.

Args:
    config_path: Path to save the configuration file. If None, uses the path
                 from which the configuration was loaded.
                 
Returns:
    bool: True if configuration was saved successfully, False otherwise.

### load_from_env

```python
def load_from_env(self)
```

Load configuration values from environment variables.

Environment variables take precedence over values in the config file.
The naming convention is NANOBRAIN_SECTION_KEY, e.g., NANOBRAIN_API_KEYS_OPENAI.

### get

```python
def get(self, path: Union[str, list], default: Any) -> Any
```

Get a configuration value by path.

Args:
    path: Path to the configuration value, either as a dot-separated string
         or a list of keys.
    default: Default value to return if the path is not found.
    
Returns:
    The configuration value, or the default if not found.

### set

```python
def set(self, path: Union[str, list], value: Any)
```

Set a configuration value by path.

Args:
    path: Path to the configuration value, either as a dot-separated string
         or a list of keys.
    value: Value to set.

### get_api_key

```python
def get_api_key(self, provider: str) -> Optional[str]
```

Get an API key for a specific provider.

Args:
    provider: The provider name (e.g., 'openai', 'anthropic').
    
Returns:
    The API key, or None if not found.

### set_api_key

```python
def set_api_key(self, provider: str, key: str)
```

Set an API key for a specific provider.

Args:
    provider: The provider name (e.g., 'openai', 'anthropic').
    key: The API key.

### setup_environment

```python
def setup_environment(self)
```

Set up the environment based on the configuration.

This includes setting environment variables for API keys and configuring logging.

### config

```python
def config(self) -> Dict[str, Any]
```

Get the entire configuration dictionary.

Returns:
    The configuration dictionary.

### config_path

```python
def config_path(self) -> Optional[str]
```

Get the path to the configuration file.

Returns:
    The path to the configuration file, or None if not loaded from a file.

