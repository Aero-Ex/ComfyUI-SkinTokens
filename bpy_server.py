import os
import sys

# Ensure the current directory and user site-packages are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
import site
user_site = site.getusersitepackages()

for path in [current_dir, user_site]:
    if path and path not in sys.path:
        sys.path.insert(0, path)

from src.server.bpy_server import run

def main():
    run()

if __name__ == "__main__":
    main()