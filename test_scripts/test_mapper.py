import os
import sys

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rig_package.parser.bpy import BpyParser
from src.rig_package.info.mixamo_mapper import map_asset_to_mixamo

def test_mapper(fbx_path):
    print(f"Loading FBX: {fbx_path}...")
    try:
        asset = BpyParser.load(filepath=fbx_path)
    except Exception as e:
        print(f"Failed to load FBX. Error: {e}")
        return

    joints = asset.joints
    parents = asset.parents
    original_names = asset.joint_names

    if joints is None or parents is None:
        print("Error: The loaded FBX does not contain joints or parents.")
        return

    print("\n--- Running Mixamo Mapper ---")
    mapped_names = map_asset_to_mixamo(joints, parents)

    print(f"{'Original Name':<20} | {'Mapped Name':<30} | {'Parent Index':<15}")
    print("-" * 70)
    for i in range(len(joints)):
        orig = original_names[i] if original_names else f"bone_{i}"
        mapped = mapped_names[i]
        parent_idx = parents[i]
        print(f"{orig:<20} | {mapped:<30} | {parent_idx:<15}")

    asset.joint_names = mapped_names
    out_path = "/home/aero/Downloads/Mixamo/res_2_mixamo.fbx"
    print(f"\nExporting to {out_path}...")
    try:
        BpyParser.export_asset(asset, out_path)
        print("Export successful!")
    except Exception as e:
        print(f"Failed to export. Error: {e}")

if __name__ == "__main__":
    test_file = "/home/aero/Downloads/Mixamo/res_2.fbx"
    test_mapper(test_file)
