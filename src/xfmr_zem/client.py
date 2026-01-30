
from typing import Dict, Any, List
import yaml
from pathlib import Path
from zenml import pipeline
from .zenml_wrapper import mcp_generic_step
import os
import sys

class PipelineClient:
    """
    Client to run Zem pipelines using MCP servers and ZenML orchestration.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.server_configs = self._load_server_configs()

    def _load_config(self, path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_server_configs(self) -> Dict[str, Any]:
        # Helper to locate server definitions based on the 'servers' block in config
        # Similar to UltraRAG's logic
        servers = self.config.get("servers", {})
        configs = {}
        for name, path_str in servers.items():
            # Assume local python file path for now as per UltraRAG example
            # In real implementation we'd parse server.yaml inside that path
            # But for simplicity, let's assume we invoke the python script directly
            abs_path = (self.config_path.parent / path_str).resolve()
            # If path points to a dir, look for src/*.py or similar. 
            # For now, let's assume the user points to the python file or we find it.
            # Simplified for MVP: path points to the python file implementing the server.
             
            env = os.environ.copy()
            # Inject src directory into PYTHONPATH so server can import xfmr_zem
            src_path = str(Path(__file__).parent.parent.resolve())
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path

            configs[name] = {
                "command": sys.executable, # Use current python interpreter
                "args": [str(abs_path)],
                "env": env
            }
        return configs

    def run(self):
        """Build and run the ZenML pipeline."""
        
        pipeline_steps = self.config.get("pipeline", [])
        server_configs = self.server_configs
        
        pipeline_name = self.config.get("name", "dynamic_generated_pipeline")

        @pipeline(name=pipeline_name, enable_cache=False)
        def dynamic_generated_pipeline():
            # Iterate through steps defined in YAML
            # Data flow tracking is simplified here (linear)
            last_output = {} 
            
            for step_def in pipeline_steps:
                srv = ""
                tool = ""
                tool_args = {}

                if isinstance(step_def, str):
                    # Format: "server.tool"
                    try:
                        srv, tool = step_def.split(".")
                    except ValueError:
                        print(f"Invalid step format: {step_def}")
                        continue
                elif isinstance(step_def, dict):
                    # Format: {"server.tool": {"input": {...}}}
                    key = list(step_def.keys())[0]
                    try:
                        srv, tool = key.split(".")
                        tool_args = step_def[key].get("input", {}) or {}
                    except ValueError:
                        print(f"Invalid step format: {step_def}")
                        continue
                
                if srv and tool:
                    # Dynamically create a step name for better visualization
                    from zenml import step as zenml_step
                    dynamic_step = zenml_step(mcp_generic_step.entrypoint, name=f"{srv}_{tool}")
                    step_output = dynamic_step(
                        server_name=srv,
                        tool_name=tool,
                        server_config=server_configs.get(srv, {}),
                        tool_args=tool_args,
                        previous_output=last_output
                    )
                    last_output = step_output
            
            return last_output

        # Run the pipeline
        p = dynamic_generated_pipeline()
        return p
        # p.run() - dynamic_generated_pipeline() already runs/submits the pipeline
