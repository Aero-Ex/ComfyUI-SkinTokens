import os
import sys
import subprocess
import shutil
import requests
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
    if os.name == 'nt':
        patterns = ["**/python/bin/python.exe", "python/bin/python.exe", "**/python.exe"]
    else:
        patterns = ["**/bin/python3*", "bin/python3*", "**/python3*"]
    
    for pattern in patterns:
        matches = list(root_dir.glob(pattern))
        # Filter out configuration scripts and directories
        matches = [m for m in matches if not str(m).endswith("-config") and m.is_file()]
        if not matches: continue
        
        # Prioritize exact names like 'python3' or 'python.exe'
        exact_names = ["python", "python3", "python.exe"]
        for name in exact_names:
            for m in matches:
                if m.name == name: return str(m)
        
        return str(matches[-1])
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
    final_blender = shutil.which("blender")
    
    # Common Windows installation paths as fallbacks
    if not final_blender and os.name == 'nt':
        common_paths = [
            "C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.1\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                final_blender = p
                break
    
    if not final_blender:
        print("WARNING: 'blender' executable not found in system PATH.")
        print("         The standalone 3D server setup will be skipped.")
        return

    print(f"Blender found at: {final_blender}")
    py_exe = find_blender_python(final_blender)
    if not py_exe:
        print(f"ERROR: Could not find internal Python in Blender path: {final_blender}")
        return

    print(f"Found Blender Python: {py_exe}")
    print(f"Installing server dependencies: {', '.join(BLENDER_DEPS)}...")
    try:
        subprocess.run([py_exe, "-m", "pip", "install"] + BLENDER_DEPS, check=True)
        print("SUCCESS: Blender server dependencies installed.")
    except Exception as e:
        print(f"ERROR: Failed to install Blender dependencies: {e}")

def install_flash_attn_section():
    print("\n--- Phase 2: Flash Attention Setup (Pre-built) ---")
    try:
        import torch
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        cuda_ver = f"cu{torch.version.cuda.replace('.', '')}" if torch.cuda.is_available() else "nocuda"
        torch_raw = torch.__version__.split('+')[0].split('.')
        torch_short = f"torch{torch_raw[0]}{torch_raw[1]}"
        torch_dotted = f"torch{torch_raw[0]}.{torch_raw[1]}"
        torch_full = torch.__version__.split('+')[0]
    except ImportError:
        print("Error: Torch must be installed first to detect the correct Flash Attention wheel.")
        return

    print(f"Detected: {py_ver}, {cuda_ver}, {torch_full}")

    repos = [
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/",
        "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/",
        "https://pozzettiandrea.github.io/cuda-wheels/flash-attn/"
    ]
    
    patterns = [
        f"flash_attn-2.8.3%2B{cuda_ver}{torch_dotted}-{py_ver}-{py_ver}",
        f"flash_attn-2.8.3+{cuda_ver}{torch_dotted}-{py_ver}-{py_ver}",
        f"flash_attn-2.8.3+{cuda_ver}torch{torch_full}cxx11abiFALSE-{py_ver}-{py_ver}",
        f"flash_attn-2.8.3+{cuda_ver}{torch_short}-{py_ver}-{py_ver}"
    ]

    possible_urls = []
    for base_url in repos:
        for p in patterns:
            tag = "win_amd64.whl" if os.name == 'nt' else "manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
            possible_urls.append(f"{base_url}{p}-{tag}")
            if os.name != 'nt':
                possible_urls.append(f"{base_url}{p}-linux_x86_64.whl")

    full_url = None
    for url in possible_urls:
        try:
            r = requests.get(url, timeout=5, allow_redirects=True, stream=True)
            if r.status_code == 200:
                full_url = url
                break
        except: continue

    if full_url:
        print(f"Target Wheel: {full_url}")
        print("Proceeding with automatic installation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", full_url], check=True)
            print("SUCCESS: Flash Attention installed.")
        except Exception as e:
            print(f"ERROR: Flash Attention installation failed: {e}")
    else:
        print("NOTICE: No matching pre-built wheel found for your configuration.")

def main():
    print("=== SkinTokens: Complete Installation Script ===")
    install_core_section()
    install_blender_section()
    install_flash_attn_section()
    print("\nInstallation process finished.")

if __name__ == "__main__":
    main()
