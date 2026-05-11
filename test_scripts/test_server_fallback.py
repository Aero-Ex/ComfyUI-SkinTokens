import builtins
real_import = builtins.__import__
def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'bpy':
        raise ImportError("No module named 'bpy'")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = mock_import

import time
import requests
from demo import start_bpy_server
from src.server.spec import BPY_PORT

print("Starting server...")
proc = start_bpy_server()
print("Waiting for server to boot...")
time.sleep(5)

try:
    print("Pinging server...")
    resp = requests.get(f"http://127.0.0.1:{BPY_PORT}/ping")
    print(f"Response: {resp.text}")
    if resp.text == "pong":
        print("SUCCESS! Headless Blender server works!")
except Exception as e:
    print(f"FAILURE: {e}")
finally:
    proc.terminate()
