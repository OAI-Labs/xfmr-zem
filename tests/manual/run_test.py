
from xfmr_zem.client import PipelineClient
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.getcwd(), "src"))

def main():
    print("Initializing Pipeline Client...")
    client = PipelineClient("tests/manual/dummy_config.yaml")
    
    print("Running Pipeline...")
    client.run()
    print("Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()
