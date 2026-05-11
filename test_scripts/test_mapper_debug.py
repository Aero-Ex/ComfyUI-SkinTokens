import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.rig_package.parser.bpy import BpyParser
import numpy as np

asset = BpyParser.load(filepath="/home/aero/Downloads/Mixamo/res_2.fbx")
joints = asset.joints
parents = asset.parents

rc = [i for i, p in enumerate(parents) if p == 0]
print("Root (0) children:", rc)
for c in rc:
    vec = joints[c] - joints[0]
    print(f"Child {c}: vector={vec}, length={np.linalg.norm(vec):.3f}")

