# DirectoryTracer

Tracks and provides the path of a class relative to the framework package root.

Biological analogy: Place cells in the hippocampus that encode spatial location.
Justification: Just as place cells allow an organism to know its location in physical space,
this class allows components to know their location in the codebase's structure.

## Methods

### get_relative_path

```python
def get_relative_path(self) -> str
```

Returns the saved relative path.

### get_absolute_path

```python
def get_absolute_path(self) -> str
```

Returns the absolute path by finding the package root and combining with relative path.

Biological analogy: Integration of egocentric and allocentric reference frames in navigation.
Justification: Similar to how the brain integrates relative positional information with
absolute map-like representations to determine precise locations.

