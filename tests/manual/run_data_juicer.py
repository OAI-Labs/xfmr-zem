import os
import sys
import json
from xfmr_zem.client import PipelineClient

def main():
    print("Initializing Pipeline Client for DataJuicer...")
    
    # Clear old error log
    if os.path.exists("/tmp/zenml_error.log"):
        os.remove("/tmp/zenml_error.log")
        
    # Use absolute path
    config_path = os.path.abspath("tests/manual/data_juicer_config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
        
    client = PipelineClient(config_path)
    
    print("Running Pipeline...")
    try:
        run_response = client.run()
        print(f"\nPipeline Run Finished: {run_response.name}")
        print(f"Status: {run_response.status}")
        
        # Check if we can get the output of the last step
        # In ZenML 0.61+, we can access steps via run_response.steps
        last_step_name = "dj_text_length_filter"
        if last_step_name in run_response.steps:
            step = run_response.steps[last_step_name]
            # Print output metadata if available
            print(f"Last step outputs: {list(step.outputs.keys())}")
        
        print("\nPipeline Finished Successfully!")
    except Exception as e:
        print(f"\nPipeline Failed with Error: {e}")
        # Check error log if exists
        if os.path.exists("/tmp/zenml_error.log"):
            print("\nError Log from /tmp/zenml_error.log:")
            with open("/tmp/zenml_error.log", "r") as f:
                print(f.read())

if __name__ == "__main__":
    main()
