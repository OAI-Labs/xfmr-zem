
from xfmr_zem.client import PipelineClient
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.getcwd(), "src"))

def main():
    print("Initializing Pipeline Client for NeMo...")
    # Use absolute path to avoid ambiguity or relative from CWD
    config_path = os.path.abspath("tests/manual/nemo_config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    client = PipelineClient(config_path)
    
    print("Running Pipeline...")
    # client.run() # No need to call run explicitly as __init__ builds it? 
    # Wait, client.run() builds and submits.
    client.run()
    print("Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()
