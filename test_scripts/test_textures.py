import os
import sys
import logging
from pathlib import Path

# Add current directory to sys.path to find src
sys.path.append(os.getcwd())

try:
    import bpy
    import numpy as np
    from src.rig_package.parser.bpy import BpyParser
    from src.rig_package.info.asset import Asset
except ImportError as e:
    print(f"Error: Could not import required modules. Make sure you are running this in the correct venv. {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)

def test_texture_preservation(input_path, output_format=".glb"):
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    output_path = os.path.join(os.getcwd(), f"test_output{output_format}")
    
    print(f"\n--- Testing Texture Preservation ---")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # 1. Load the asset
    print("\n[Step 1] Loading mesh...")
    asset = BpyParser.load(filepath=input_path)
    
    # Check what's in Blender now
    mats = list(bpy.data.materials)
    imgs = list(bpy.data.images)
    print(f"Materials found in scene: {[m.name for m in mats]}")
    print(f"Images found in scene: {[i.name for i in imgs]}")
    
    if not mats:
        print("WARNING: No materials found after loading! Texture preservation will fail.")
    if not imgs:
        print("WARNING: No images found after loading! Texture preservation will fail.")

    # 2. Export the asset
    print(f"\n[Step 2] Exporting to {output_format}...")
    # Simulate a transfer rigging output (using the loaded asset)
    try:
        BpyParser.export(asset, output_path, use_origin=True)
        print(f"Export successful: {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Verify output
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"\n[Step 3] Output verification:")
        print(f"File size: {size / 1024:.2f} KB")
        if size < 50 * 1024: # Random threshold
             print("WARNING: File size seems small. Textures might not be embedded.")
    else:
        print("\n[Step 3] Error: Output file was not created.")

if __name__ == "__main__":
    # Change this to a path that exists on your system for testing
    test_file = "/home/aero/comfy/ComfyUI/input/3d/Baby Groot.fbx"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    test_texture_preservation(test_file, ".glb")
    test_texture_preservation(test_file, ".fbx")
    test_texture_preservation(test_file, ".obj")
