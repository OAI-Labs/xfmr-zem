
from typing import Any, Callable, Dict, List, Optional, Union
import yaml
from pathlib import Path
from fastmcp import FastMCP
import inspect

class ZemServer(FastMCP):
    """
    Base class for Zem MCP Servers.
    Extends FastMCP to support parameter loading and standardized tool registration.
    """
    
    def __init__(
        self,
        name: str,
        Dependencies: Optional[List[str]] = None,
        parameter_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.parameter_file = parameter_file
        self.parameters = {}
        if parameter_file:
            self.load_parameters(parameter_file)

    def load_parameters(self, file_path: str) -> Dict[str, Any]:
        """Load parameters from YAML file."""
        path = Path(file_path)
        if path.exists():
            with open(path, "r") as f:
                self.parameters = yaml.safe_load(f) or {}
            return self.parameters
        return {}

    # Removed custom tool decorator to fix multiple values for argument 'name' error
    # Inherit directly from FastMCP.tool

    def run(self, transport: str = "stdio"):
        """Run the server."""
        super().run(transport=transport, show_banner=False)
