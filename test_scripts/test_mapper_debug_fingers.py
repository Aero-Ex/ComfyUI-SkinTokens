import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.rig_package.parser.bpy import BpyParser
import numpy as np

asset = BpyParser.load(filepath="/home/aero/Downloads/Mixamo/res_2.fbx")
joints = asset.joints
parents = asset.parents

# bone_8 is L_Wrist based on the previous output
c_8 = [i for i, p in enumerate(parents) if p == 8]
print("L_Wrist (8) children:", c_8)
for c in c_8:
    vec = joints[c] - joints[8]
    print(f"Finger {c}: vector={vec}")

