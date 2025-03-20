import os
from pathlib import Path
from src.interfaces import IDirectoryTracer


class DirectoryTracer(IDirectoryTracer):
    """
    Tracks and provides the directory of a class relative to the framework package root.
    
    Biological analogy: Place cells in the hippocampus that encode spatial location.
    Justification: Just as place cells allow an organism to know its location in physical space,
    this class allows components to know their location in the codebase's structure.
    """
    def __init__(self, module_name: str):
        pathes = module_name.split('.')
        if len(pathes) > 1:
            self.relative_directory_location = os.path.join(*pathes[:-1])
        else:
            self.relative_directory_location = '.'
        self.relative_path = os.path.join(*pathes)

    def get_relative_path(self) -> str:
        return self.relative_path

    def get_absolute_path(self) -> str:
        package_root = Path(__file__).parent.parent
        return os.path.join(str(package_root), self.relative_path) 

    def get_relative_directory_location(self) -> str:
        """Returns the saved relative path."""
        return self.relative_path
    
    def get_absolute_directory_location(self) -> str:
        """
        Returns the absolute path by finding the package root and combining with relative path.
        
        Biological analogy: Integration of egocentric and allocentric reference frames in navigation.
        Justification: Similar to how the brain integrates relative positional information with
        absolute map-like representations to determine precise locations.
        """
        # Find the package root
        package_root = Path(__file__).parent.parent
        return os.path.join(str(package_root), self.relative_directory_location) 