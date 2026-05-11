import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path to allow imports when run as standalone
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.rig_package.parser.bpy import BpyParser

MIXAMO_NAMES = {
    "Pelvis": "mixamorig:Hips",
    "L_Hip": "mixamorig:LeftUpLeg",
    "L_Knee": "mixamorig:LeftLeg",
    "L_Ankle": "mixamorig:LeftFoot",
    "L_Foot": "mixamorig:LeftToeBase",
    "L_Toe": "mixamorig:LeftToeBase",
    "R_Hip": "mixamorig:RightUpLeg",
    "R_Knee": "mixamorig:RightLeg",
    "R_Ankle": "mixamorig:RightFoot",
    "R_Foot": "mixamorig:RightToeBase",
    "R_Toe": "mixamorig:RightToeBase",
    "Spine1": "mixamorig:Spine",
    "Spine2": "mixamorig:Spine1",
    "Spine3": "mixamorig:Spine2",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "HeadTop": "mixamorig:HeadTop_End",
    "L_Collar": "mixamorig:LeftShoulder",
    "L_Shoulder": "mixamorig:LeftArm",
    "L_Elbow": "mixamorig:LeftForeArm",
    "L_Wrist": "mixamorig:LeftHand",
    "R_Collar": "mixamorig:RightShoulder",
    "R_Shoulder": "mixamorig:RightArm",
    "R_Elbow": "mixamorig:RightForeArm",
    "R_Wrist": "mixamorig:RightHand",
}

for prefix, m_prefix in [("L", "LeftHand"), ("R", "RightHand")]:
    for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        for i in [1, 2, 3]:
            MIXAMO_NAMES[f"{prefix}_{f}{i}"] = f"mixamorig:{m_prefix}{f}{i}"

def map_asset_to_mixamo(joints: np.ndarray, parents: np.ndarray) -> List[str]:
    """
    Takes a generated skeleton's joints and parents and returns a list
    of Mixamo bone names corresponding to each index.
    Unmatched bones will keep a bone_X format.
    """
    J = len(joints)
    labels = [f"bone_{i}" for i in range(J)]
    
    children = {i: [] for i in range(J)}
    for i in range(J):
        if parents[i] != -1:
            children[parents[i]].append(i)
            
    root_idx = -1
    for i in range(J):
        if parents[i] == -1:
            root_idx = i
            break
            
    if root_idx == -1:
        return labels
        
    labels[root_idx] = "Pelvis"
    
    rc = children[root_idx]
    if not rc: return labels
    
    # Spine is the child pointing highest (max Z)
    spine_idx = max(rc, key=lambda c: joints[c][2])
    hips = [c for c in rc if c != spine_idx]
    
    l_hip_idx, r_hip_idx = -1, -1
    if len(hips) >= 2:
        # Sort by X. Assuming +X is Left (Standard T-Pose)
        hips.sort(key=lambda c: joints[c][0])
        r_hip_idx = hips[0]
        l_hip_idx = hips[-1]
    elif len(hips) == 1:
        if joints[hips[0]][0] > 0:
            l_hip_idx = hips[0]
        else:
            r_hip_idx = hips[0]
            
    def label_chain(start, child_map, names):
        curr = start
        for name in names:
            if curr == -1: break
            labels[curr] = name
            if not child_map[curr]: break
            curr = child_map[curr][0]
            
    if l_hip_idx != -1:
        label_chain(l_hip_idx, children, ["L_Hip", "L_Knee", "L_Ankle", "L_Foot", "L_Toe"])
    if r_hip_idx != -1:
        label_chain(r_hip_idx, children, ["R_Hip", "R_Knee", "R_Ankle", "R_Foot", "R_Toe"])
        
    curr = spine_idx
    spine_count = 1
    while curr != -1:
        c = children[curr]
        if len(c) == 1:
            labels[curr] = f"Spine{spine_count}"
            spine_count += 1
            curr = c[0]
        elif len(c) >= 3:
            labels[curr] = f"Spine{spine_count}"
            
            # Neck is highest Z
            neck_idx = max(c, key=lambda x: joints[x][2])
            collars = [x for x in c if x != neck_idx]
            collars.sort(key=lambda x: joints[x][0])
            r_collar_idx = collars[0] if len(collars) > 0 else -1
            l_collar_idx = collars[-1] if len(collars) > 1 else -1
            
            label_chain(neck_idx, children, ["Neck", "Head", "HeadTop"])
            
            def label_fingers(wrist_idx, prefix):
                fingers = children[wrist_idx]
                if not fingers: return
                # Thumb is the shortest vector from wrist
                thumb_idx = min(fingers, key=lambda f: np.linalg.norm(joints[f] - joints[wrist_idx]))
                other_fingers = [f for f in fingers if f != thumb_idx]
                # Sort other 4 fingers by Y (forward to back, ascending)
                # Lowest Y is front (Index), Highest Y is back (Pinky)
                other_fingers.sort(key=lambda f: joints[f][1], reverse=False)
                
                label_chain(thumb_idx, children, [f"{prefix}_Thumb1", f"{prefix}_Thumb2", f"{prefix}_Thumb3"])
                finger_names = ["Index", "Middle", "Ring", "Pinky"]
                for i, f_idx in enumerate(other_fingers):
                    if i >= 4: break
                    name = finger_names[i]
                    label_chain(f_idx, children, [f"{prefix}_{name}1", f"{prefix}_{name}2", f"{prefix}_{name}3"])
            
            if l_collar_idx != -1:
                label_chain(l_collar_idx, children, ["L_Collar", "L_Shoulder", "L_Elbow", "L_Wrist"])
                l_wrist = -1
                for w in range(J):
                    if labels[w] == "L_Wrist": l_wrist = w; break
                if l_wrist != -1:
                    label_fingers(l_wrist, "L")
                    
            if r_collar_idx != -1:
                label_chain(r_collar_idx, children, ["R_Collar", "R_Shoulder", "R_Elbow", "R_Wrist"])
                r_wrist = -1
                for w in range(J):
                    if labels[w] == "R_Wrist": r_wrist = w; break
                if r_wrist != -1:
                    label_fingers(r_wrist, "R")
            
            break
        else:
            labels[curr] = f"Spine{spine_count}"
            break

    # Convert semantic labels to Mixamo
    final_names = []
    for label in labels:
        if label in MIXAMO_NAMES:
            final_names.append(MIXAMO_NAMES[label])
        else:
            final_names.append(label)
            
    return final_names

def main():
    parser = argparse.ArgumentParser(description="Standalone Mixamo Bone Mapper")
    parser.add_argument("--input", required=True, help="Path to input 3D file (fbx, glb, obj)")
    parser.add_argument("--output", help="Path to output file (default: input_mixamo.ext)")
    parser.add_argument("--format", choices=["glb", "fbx", "obj"], help="Force output format")
    
    args = parser.parse_args()
    
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"Error: File {in_path} does not exist.")
        return

    # Determine output path
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        ext = args.format if args.format else in_path.suffix[1:]
        out_path = in_path.with_name(f"{in_path.stem}_mixamo.{ext}")

    print(f"Loading {in_path}...")
    asset = BpyParser.load(str(in_path))
    
    print("Mapping bones to Mixamo...")
    original_names = asset.joint_names.copy() if asset.joint_names is not None else [f"bone_{i}" for i in range(len(asset.joints))]
    new_names = map_asset_to_mixamo(asset.joints, asset.parents)
    asset.joint_names = new_names
    
    print("\nBone Mapping Log:")
    print("-" * 40)
    for old, new in zip(original_names, new_names):
        if old != new:
            print(f"  {old} -> {new}")
    print("-" * 40 + "\n")
    
    print(f"Exporting to {out_path}...")
    BpyParser.export(asset, str(out_path))
    print("Done!")

if __name__ == "__main__":
    main()
