
import subprocess
import sys
import os
import json

def main():
    server_script = "tests/manual/dummy_server.py"
    cmd = [sys.executable, server_script]
    
    # Inject PYTHONPATH
    env = os.environ.copy()
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"
    
    print(f"Launching server: {cmd}")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=0 # Unbuffered
    )

    try:
        # JSON-RPC Initialize Request
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", # MCP version roughly
                "capabilities": {},
                "clientInfo": {"name": "debug-client", "version": "1.0"}
            }
        }
        
        print("Sending Initialize Request...")
        msg = json.dumps(init_req) + "
"
        process.stdin.write(msg)
        process.stdin.flush()
        
        print("Waiting for response...")
        # Read line
        output = process.stdout.readline()
        print(f"Received: {output}")
        
        if output:
            print("Server is responding!")
        else:
            print("Server closed stdout unexpectedly.")
            # check stderr
            err = process.stderr.read()
            print(f"Stderr: {err}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
