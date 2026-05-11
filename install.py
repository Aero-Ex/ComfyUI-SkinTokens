import os
import sys
import subprocess
import shutil
import re
from pathlib import Path

# Dependencies required for the current environment (ComfyUI/Venv)
CORE_DEPS = [
    "dill", "python-box", "einops", "omegaconf", "lightning", "addict", 
    "fast-simplification", "trimesh", "open3d", "gradio", "bottle", "tornado"
]

# Dependencies required specifically for the Blender standalone server
BLENDER_DEPS = ["bottle", "dill", "scipy", "trimesh", "tornado"]

def find_blender_python(blender_path):
    """Finds the internal python executable relative to the blender executable."""
    blender_bin = Path(blender_path)
    root_dir = blender_bin.parent
    patterns = ["*/python/bin/python.exe", "python/bin/python.exe"] if os.name == 'nt' else ["*/python/bin/python3*", "python/bin/python3*"]
    for pattern in patterns:
        matches = list(root_dir.glob(pattern))
        if matches: return str(matches[-1])
    return None

def install_core_section():
    print("\n--- Phase 0: Core Library Setup ---")
    
    # Try to install bpy optionally
    print("Attempting to install 'bpy' (optional native support)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "bpy"], check=True)
        print("SUCCESS: Native 'bpy' installed.")
    except Exception:
        print("NOTICE: 'bpy' installation failed or is not supported for this Python version.")
        print("        The node will use the Headless Blender Server instead (this is normal).")

    print(f"Installing required core dependencies: {', '.join(CORE_DEPS)}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install"] + CORE_DEPS, check=True)
        print("SUCCESS: Core dependencies installed.")
    except Exception as e:
        print(f"ERROR: Failed to install core dependencies: {e}")

def install_blender_section():
    print("\n--- Phase 1: Blender Standalone Server Setup ---")
    blender_cmd = shutil.which("blender")
    user_input = input(f"Enter path to blender executable [Default: {blender_cmd or 'Not Found'}]: ").strip()
    final_blender = user_input if user_input else blender_cmd
    if not final_blender: return
    py_exe = find_blender_python(final_blender)
    if not py_exe: return
    print(f"Installing Blender dependencies into {py_exe}...")
    subprocess.run([py_exe, "-m", "pip", "install"] + BLENDER_DEPS)

def install_flash_attn_section():
    print("\n--- Phase 2: Flash Attention 2 Setup (Manual Wheel Matching) ---")
    try:
        import torch
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        torch_raw = torch.__version__.split('+')[0].split('.')
        torch_ver = f"torch{torch_raw[0]}{torch_raw[1]}"
        cuda_ver = f"cu{torch.version.cuda.replace('.', '')}" if torch.cuda.is_available() else "nocuda"
        os_tag = "win_amd64" if os.name == 'nt' else "manylinux"
    except ImportError:
        print("Error: Torch must be installed first to detect the correct Flash Attention wheel.")
        return

    print(f"Detected: {py_ver}, {torch_ver}, {cuda_ver}, {os_tag}")

    base_url = "https://pozzettiandrea.github.io/cuda-wheels/flash-attn/"
    target_filename = f"flash_attn-2.8.3+{cuda_ver}{torch_ver}-{py_ver}-{py_ver}"
    if os.name == 'nt':
        target_filename += "-win_amd64.whl"
    else:
        target_filename += "-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

    full_url = f"{base_url}{target_filename}"
    print(f"Target URL: {full_url}")
    
    choice = input(f"Proceed with installing this specific wheel? [y/N]: ").lower()
    if choice == 'y':
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", full_url], check=True)
            print("SUCCESS: Flash Attention installed via direct wheel link.")
        except Exception as e:
            print(f"\nFailed to install specific wheel.")

def main():
    print("=== SkinTokens: Smart Installation Script ===")
    install_core_section()
    install_blender_section()
    install_flash_attn_section()
    print("\nAll tasks finished!")

if __name__ == "__main__":
    main()
