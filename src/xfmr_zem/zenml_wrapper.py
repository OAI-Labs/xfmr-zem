
from typing import Any, Dict, Optional
from zenml import step
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import os

import subprocess
import time

# Helper to run async MCP call synchronously
def run_mcp_tool(
    command: str,
    args: list[str],
    env: Dict[str, str],
    tool_name: str,
    tool_args: Dict[str, Any]
) -> Any:
    """
    Manually run the MCP server subprocess and call the tool via JSON-RPC over stdio.
    This avoids complex async/contextlib issues with the mcp library in this environment.
    """
    cmd = [command] + args
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=0 
    )

    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "zenml-client", "version": "1.0"}
            }
        }
        process.stdin.write(json.dumps(init_req) + "\n")
        process.stdin.flush()
        
        # Read init response
        while True:
            init_resp_line = process.stdout.readline()
            if not init_resp_line:
                 err = process.stderr.read()
                 raise RuntimeError(f"Server closed connection during init. Stderr: {err}")
            
            # print(f"DEBUG: init_line: {init_resp_line.strip()}")
            if init_resp_line.strip().startswith("{"):
                try:
                    init_resp = json.loads(init_resp_line)
                    break
                except json.JSONDecodeError:
                    continue
        
        # 2. Call Tool
        call_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        process.stdin.write(json.dumps(call_req) + "\n")
        process.stdin.flush()
        
        # Read tool response
        while True:
            tool_resp_line = process.stdout.readline()
            if not tool_resp_line:
                 err = process.stderr.read()
                 raise RuntimeError(f"Server closed connection during tool call. Stderr: {err}")
            
            # print(f"DEBUG: tool_line: {tool_resp_line.strip()}")
            if tool_resp_line.strip().startswith("{"):
                try:
                    # print(f"DEBUG: Attempting to parse: {tool_resp_line.strip()[:100]}")
                    tool_resp = json.loads(tool_resp_line)
                    break
                except json.JSONDecodeError as je:
                    print(f"DEBUG: JSON parse failed for line starting with {{: {tool_resp_line.strip()[:100]} Error: {je}")
                    continue
             
        # tool_resp is now loaded correctly
        
        if "error" in tool_resp:
            raise RuntimeError(f"MCP Tool Error: {tool_resp['error']}")
            
        return tool_resp["result"]

    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except:
            process.kill()


@step
def mcp_generic_step(
    server_name: str,
    tool_name: str,
    server_config: Dict[str, Any],
    tool_args: Dict[str, Any],
    previous_output: Optional[Any] = None
) -> Any:
    """
    A generic ZenML step that executes a tool on an MCP server.
    """
    # Merge previous output into tool_args if present
    if previous_output is not None:
        print(f"[{server_name}] Received input from previous step (type: {type(previous_output)})")
        # Logic to map previous output to tool args
        # 1. If previous output is a dict, merge it (priority to manual args)
        if isinstance(previous_output, dict):
            # We want current tool_args to override previous_output keys if collision?
            # Usually strict update. 
            # But deep merge might be better. 
            # Let's do implicit merge: tool_args + previous_output
            # usage: previous output provided 'data', config provided 'threshold'
            merged = previous_output.copy()
            merged.update(tool_args)
            tool_args = merged
        else:
            # 2. If previous output is list/other, assume it's the 'data' argument
            # This is specific to our Zem data pipeline convention
            tool_args['data'] = previous_output

    command = server_config.get("command", "python")
    args = server_config.get("args", [])
    env = server_config.get("env", os.environ.copy())
    
    print(f"[{server_name}] Executing tool '{tool_name}' with args keys: {list(tool_args.keys())}")
    # print(f"[{server_name}] Command: {command} {args}")
    
    try:
        result_data = run_mcp_tool(command, args, env, tool_name, tool_args)
        
        # print(f"[{server_name}] Raw Result: {result_data}")
        
        output_data = {}
        
        # Handle standard MCP content structure
        if isinstance(result_data, dict) and "content" in result_data:
            content = result_data["content"]
            if isinstance(content, list) and len(content) > 0:
                item = content[0]
                if item.get("type") == "text":
                    text = item.get("text", "")
                    # print(f"[{server_name}] Text Output: {text[:200]}...") # Truncate log
                    try:
                        output_data = json.loads(text)
                    except:
                        output_data = {"raw_output": text}
        else:
             # Fallback if tool returns raw data (unlikely strict MCP but possible in custom impl)
             output_data = result_data if isinstance(result_data, dict) else {"raw": str(result_data)}
        
        if isinstance(output_data, list):
            print(f"[{server_name}] Output: {len(output_data)} items")
        elif isinstance(output_data, dict) and "data" in output_data:
            print(f"[{server_name}] Output: {len(output_data['data'])} items")

        return output_data
        
    except Exception as e:
        import traceback
        with open("/tmp/zenml_error.log", "w") as f:
            f.write(f"Error executing {server_name}.{tool_name}:\n")
            traceback.print_exc(file=f)
        raise RuntimeError(f"Failed to execute MCP tool {server_name}.{tool_name}: {e}")
