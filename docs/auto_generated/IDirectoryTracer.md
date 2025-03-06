# IDirectoryTracer

Interface for components that need to track their location in the codebase.

Biological analogy: Place cells in the hippocampus that encode spatial location.
Justification: Just as place cells allow an organism to know its location in physical space,
this interface defines how components can know their location in the codebase's structure.

## Methods

### get_relative_path

```python
def get_relative_path(self) -> str
```

Returns the relative path from the package root.

### get_absolute_path

```python
def get_absolute_path(self) -> str
```

Returns the absolute path in the filesystem.

