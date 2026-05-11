from collections import defaultdict
from numpy import ndarray
from typing import Optional, Tuple

try:
    import bpy # type: ignore
except ImportError:
    bpy = None
import logging
import numpy as np
import os
import sys
import trimesh

from .abstract import AbstractParser
from ..info.asset import Asset
try:
    from mathutils import Vector, Matrix # type: ignore
except ImportError:
    Vector = Matrix = any

class BpyParser(AbstractParser):
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> Asset:
        clean_bpy()
        load(filepath=filepath, **kwargs)
        collection = bpy.data.collections.get("glTF_not_exported")
        if collection is not None:
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        armature = get_armature()
        if armature is None:
            bones = None
            joint_names = None
            parents = None
            lengths = None
            matrix_world = np.eye(4)
            matrix_local = None
            matrix_basis = None
            armature_name = None
        else:
            bones = armature.pose.bones # list of PoseBone
            joint_names = [b.name for b in bones]
            parents = []
            lengths = []
            matrix_world = np.array(armature.matrix_world)
            obj = armature.parent
            while obj is not None:
                matrix_world = np.array(obj.matrix_world) @ matrix_world
                obj = obj.parent
            
            matrix_local = []
            for pbone in bones:
                matrix_local.append(np.array(pbone.bone.matrix_local))
                parents.append(joint_names.index(pbone.parent.name) if pbone.parent is not None else -1)
                lengths.append(pbone.bone.length)
            matrix_local = np.stack(matrix_local, axis=0)
            parents = np.array(parents, dtype=np.int32)
            lengths = np.array(lengths, dtype=np.float32)
            
            matrix_basis = get_matrix_basis(bones=bones)
            armature_name = armature.name
        mesh_dict = extract_mesh(bones=bones)
        
        return Asset(
            vertices=mesh_dict['vertices'],
            faces=mesh_dict['faces'],
            vertex_normals=mesh_dict['vertex_normals'],
            face_normals=mesh_dict['face_normals'],
            vertex_bias=mesh_dict['vertex_bias'],
            face_bias=mesh_dict['face_bias'],
            mesh_names=mesh_dict['mesh_names'],
            joint_names=joint_names,
            parents=parents,
            lengths=lengths,
            matrix_world=matrix_world,
            matrix_local=matrix_local,
            matrix_basis=matrix_basis,
            armature_name=armature_name,
            skin=mesh_dict['skin'],
        )
    
    @classmethod
    def export(cls, asset: Asset, filepath: str, **kwargs):
        """
        If export obj, kwargs:
            precision: int=6, number of decimal places for vertex coordinates
        
        Otherwise, export fbx/glb/gltf using bpy, kwargs:
            extrude_scale: float=0.5, if there is no tails in asset, first calculate the average length between parents and sons, then the length of leaf bone is l*extrude_scale. Otherwise do not affect final results.
            
            connect_tail_to_unique_child: bool=False, if True, the tail of a bone with only one child will be exactly at the head of its child.
            
            extrude_from_parent: bool=False, if True, the orientation of the leaf bone will be the same as its parent.
            
            group_per_vertex: int=-1, number of the largest weights to keep for each vertex. -1 means keep all.
            
            add_root: bool=False, if True, add a root bone at (0, 0, 0).
            
            do_not_normalize: bool=False, if True, do not normalize the skinning weights.
            
            collection_name: str='new_collection', name of the new collection to store objects.
            
            add_leaf_bones: bool=False, if True, add a leaf bone at the end of each bone.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.ply':
            cls.export_ply(asset, filepath, **kwargs)
        elif ext == '.obj' and kwargs.get('use_manual_obj', False):
            cls.export_obj(asset, filepath, **kwargs)
        else:
            cls.export_asset(asset, filepath, **kwargs)
    
    @classmethod
    def export_obj(
        cls,
        asset: Asset,
        filepath: str,
        precision: int=6,
        use_pc: bool=False,
        use_normal: bool=False,
        use_skeleton: bool=False,
        normal_size: float=0.01,
    ):
        """
        Export the asset as an .obj file. This will ignore skeleton and skinning.
        
        Args:
            use_normal: export normals
            
            use_skeleton: export skeleton
        """
        asset._build_bias()
        if asset.vertices is None or asset.vertex_bias is None:
            raise ValueError("do not have vertices or vertex_bias")
        if use_normal and asset.vertex_normals is None:
            raise ValueError("use_normal is True but do not have vertex_normals")
        if not filepath.lower().endswith('.obj'):
            filepath += ".obj"
        faces = asset.faces
        mesh_names = asset.mesh_names
        if mesh_names is None:
            mesh_names = [f"mesh_{i}" for i in range(asset.P)]
        cls._safe_make_dir(filepath)
        file = open(filepath, 'w')
        lines = []
        tot = 0
        if use_skeleton:
            raise NotImplementedError()
        for i, mesh_name in enumerate(mesh_names):
            lines.append(f'o {mesh_name}\n')
            if use_normal:
                s = asset.get_vertex_slice(i)
                for v, n in zip(asset.vertices[s], asset.vertex_normals[s]): # type: ignore
                    vv = v + n * normal_size
                    lines.append(f'v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n')
                    lines.append(f'v {vv[0]:.{precision}f} {vv[2]:.{precision}f} {-vv[1]:.{precision}f}\n')
                    lines.append(f'v {vv[0]:.{precision}f} {vv[2]:.{precision}f} {-vv[1]+0.000001:.{precision}f}\n')
                    lines.append(f"f {tot+1} {tot+2} {tot+3}\n")
                    tot += 3
            else:
                for v in asset.vertices[asset.get_vertex_slice(i)]:
                    lines.append(f'v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n')
                if faces is not None and use_pc == False:
                    for f in faces[asset.get_face_slice(i)]:
                        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        file.writelines(lines)
        file.close()
    
    @classmethod
    def export_ply(
        cls,
        asset: Asset,
        filepath: str,
        use_pc: bool=False,
        render_skin_id: Optional[int]=None,
    ):
        """
        Export the asset as an .ply file. This will ignore skeleton and skinning.
        """
        import open3d as o3d
        asset._build_bias()
        if asset.vertices is None or asset.vertex_bias is None:
            raise ValueError("do not have vertices or vertex_bias")
        if not filepath.lower().endswith('.ply'):
            filepath += ".ply"
        faces = asset.faces
        if use_pc:
            faces = None
        mesh_names = asset.mesh_names
        if mesh_names is None:
            mesh_names = [f"mesh_{i}" for i in range(asset.P)]
        cls._safe_make_dir(filepath)
        
        if render_skin_id is not None:
            if asset.skin is None:
                raise ValueError("render_skin_id is not None, but skin of asset is None")
            colors = np.stack([
                asset.skin[:, render_skin_id],
                np.zeros(asset.N),
                1-asset.skin[:, render_skin_id],
            ], axis=1)
        else:
            colors = None
        if faces is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(asset.vertices)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(filepath, pcd)
        else:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(asset.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_triangle_mesh(filepath, mesh)
    
    @classmethod
    def export_asset(cls, asset: Asset, filepath: str, **kwargs):
        use_origin = kwargs.pop('use_origin', False) if 'use_origin' in kwargs else False
        # Root Cause Fix: DO NOT clean_bpy here. 
        # We need to preserve the images loaded during the 'load()' phase.
        make_asset(asset=asset, **kwargs)
        cls._safe_make_dir(filepath)
        
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()[1:]
        if ext == 'fbx':
            filepath = os.path.abspath(filepath)
            output_dir = os.path.dirname(filepath)
            model_basename = os.path.basename(filepath).replace(".fbx", "")
            
            # Create a dedicated .fbm folder for organization
            fbm_dir = os.path.join(output_dir, f"{model_basename}.fbm")
            os.makedirs(fbm_dir, exist_ok=True)
            print(f"[FBX EXPORT] Organizing textures into: {fbm_dir}")
            
            # Find ALL images, including those hidden in materials
            images_to_save = set(bpy.data.images)
            for mat in bpy.data.materials:
                if mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            images_to_save.add(node.image)
            
            for img in images_to_save:
                try:
                    # Use a clean name inside the .fbm folder
                    clean_img_name = img.name.replace(".", "_").lower()
                    fallback_filename = f"{clean_img_name}.png"
                    img_path = os.path.join(fbm_dir, fallback_filename)
                    
                    if not img.has_data:
                        try:
                            _ = img.pixels[0]
                        except:
                            continue

                    img.file_format = 'PNG'
                    try:
                        img.save_render(img_path)
                    except:
                        img.save(filepath=img_path)
                        
                    if os.path.exists(img_path):
                        print(f"[FBX EXPORT] SUCCESS: Saved to .fbm/{fallback_filename}")
                except Exception as e:
                    print(f"[FBX EXPORT] Failed to save {img.name}: {e}")
            sys.stdout.flush()

            if asset.joints is not None and asset.matrix_basis is not None:
                logging.warning("Exporting animation, but fbx format is deprecated because the rest pose will not be exported in bpy4.2. Use glb/gltf format instead. See: https://blender.stackexchange.com/questions/273398/blender-export-fbx-lose-the-origin-rest-pose.")
            bpy.ops.export_scene.fbx(filepath=filepath, check_existing=False, add_leaf_bones=kwargs.get('add_leaf_bones', False), path_mode='COPY', embed_textures=True, mesh_smooth_type="FACE")
        elif ext == 'glb' or ext == 'gltf':
            bpy.ops.export_scene.gltf(filepath=filepath)
        elif ext == 'obj':
            if hasattr(bpy.ops.wm, "obj_export"):
                bpy.ops.wm.obj_export(filepath=filepath)
            else:
                bpy.ops.export_scene.obj(filepath=filepath, use_selection=False)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    @classmethod
    def _safe_make_dir(cls, path: str):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)

def clean_bpy():
    """Clean all the data in bpy."""
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    data_types = [
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.cameras,
        bpy.data.collections,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.objects,
        bpy.data.worlds,
        bpy.data.node_groups,
        # Root Cause Fix: Do NOT clean images/textures here.
        # We need them to persist from the GLB load to the FBX export.
    ]
    for data_collection in data_types:
        for item in data_collection:
            data_collection.remove(item)

def load(filepath: str, **kwargs):
    """Load a 3D file into bpy."""
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()[1:]
    
    if not os.path.exists(filepath):
        raise RuntimeError(f"file does not exist: {filepath}")
    
    if ext == "obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(
            filepath=filepath,
            ignore_leaf_bones=kwargs.get('ignore_leaf_bones', False),
            use_image_search=kwargs.get('use_image_search', True),
        )
    elif ext == "glb" or ext == "gltf":
        bpy.ops.import_scene.gltf(filepath=filepath, import_pack_images=kwargs.get('import_pack_images', False))
    elif ext == "dae":
        bpy.ops.wm.collada_import(filepath=filepath)
    elif ext == "blend":
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    elif ext == "bvh":
        bpy.ops.import_anim.bvh(filepath=filepath)
    else:
        raise ValueError(f"unsupported type: {ext}")

def get_armature():
    """Get the armature object in the current scene."""
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if len(armatures) == 0:
        return None
    return armatures[0]

def extract_mesh(bones=None):
    """
    Extract vertices, face_normals, faces and skinning(if possible).
    """
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)
    
    index = {}
    if bones is not None:
        for (id, pbone) in enumerate(bones):
            index[pbone.name] = id
        total_bones = len(bones)
    else:
        total_bones = None
    
    mesh_names_list = []
    vertices_list = []
    faces_list = []
    skin_list = []
    vertex_bias = []
    face_bias = []
    cur_vertex_bias = 0
    cur_face_bias = 0
    for obj in meshes:
        # directly apply mesh's transformation because armature operates on the transformed mesh
        if obj.parent is not None:
            m = np.linalg.inv(np.array(obj.parent.matrix_world)) @ np.array(obj.matrix_world)
        else:
            m = np.array(obj.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        rot = matrix_world_rot
        total_vertices = len(obj.data.vertices)
        vertices = np.zeros((3, total_vertices))
        if total_bones is not None:
            skin_weight = np.zeros((total_vertices, total_bones))
        else:
            skin_weight = np.zeros((1, 1))
        obj_verts = obj.data.vertices
        obj_group_names = [g.name for g in obj.vertex_groups]
        faces = []
        normals = []
        
        for polygon in obj.data.polygons:
            edges = polygon.edge_keys
            nodes = []
            adj = {}
            for edge in edges:
                if adj.get(edge[0]) is None:
                    adj[edge[0]] = []
                adj[edge[0]].append(edge[1])
                if adj.get(edge[1]) is None:
                    adj[edge[1]] = []
                adj[edge[1]].append(edge[0])
                nodes.append(edge[0])
                nodes.append(edge[1])
            normal = polygon.normal
            nodes = list(set(sorted(nodes)))
            first = nodes[0]
            loop = []
            now = first
            vis = {}
            while True:
                loop.append(now)
                vis[now] = True
                if vis.get(adj[now][0]) is None:
                    now = adj[now][0]
                elif vis.get(adj[now][1]) is None:
                    now = adj[now][1]
                else:
                    break
            for (second, third) in zip(loop[1:], loop[2:]):
                faces.append((first, second, third))
                normals.append(rot @ normal)
        faces = np.array(faces, dtype=np.int32)
        normals = np.array(normals, dtype=np.float32)
        
        coords = np.array([v.co for v in obj_verts])
        rot_np = np.array(rot)
        coords = (rot_np @ coords.T).T + matrix_world_bias
        vertices[0:3, :coords.shape[0]] = coords.T
        
        # extract skin
        if bones is not None:
            vg_lut = {}
            for v in obj_verts:
                for g in v.groups:
                    vg_lut[(v.index, g.group)] = g.weight
            
            for bone in bones:
                if bone.name not in obj_group_names:
                    continue
                gidx = obj.vertex_groups[bone.name].index
                col = index[bone.name]
                for v in obj_verts:
                    w = vg_lut.get((v.index, gidx))
                    if w is not None:
                        skin_weight[v.index, col] = w
        
        vertices = vertices.T
        # determine the orientation of the face normal
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1-v0, v2-v0)
        dot = np.einsum("ij,ij->i", cross, normals)
        correct_faces = faces.copy()
        mask = dot < 0
        correct_faces[mask, 1], correct_faces[mask, 2] = faces[mask, 2], faces[mask, 1]
        
        mesh_names_list.append(obj.name)
        vertices_list.append(vertices)
        faces_list.append(correct_faces+cur_vertex_bias) # add bias to faces
        if total_bones is not None:
            skin_list.append(skin_weight)
        cur_vertex_bias += len(vertices)
        cur_face_bias += len(faces)
        vertex_bias.append(cur_vertex_bias)
        face_bias.append(cur_face_bias)
    
    vertices = np.vstack(vertices_list)
    faces = np.vstack(faces_list)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    
    return {
        'mesh_names': np.array(mesh_names_list),
        'vertices': vertices,
        'faces': faces,
        'face_normals': face_normals,
        'vertex_normals': vertex_normals,
        'skin': np.vstack(skin_list) if len(skin_list) > 0 else None,
        'vertex_bias': np.array(vertex_bias),
        'face_bias': np.array(face_bias),
    }

def get_matrix_basis(bones=None):
    if bones is None:
        return None
    if bpy.data.actions is not None and len(bpy.data.actions) > 0:
        action = bpy.data.actions[0]
        frames = int(action.frame_range.y - action.frame_range.x)
    else:
        return None
    
    J = len(bones)
    matrix_basis = np.zeros((frames, J, 4, 4))
    matrix_basis[...] = np.eye(4)
    for frame in range(frames):
        bpy.context.scene.frame_set(frame + 1)
        for (id, pbone) in enumerate(bones):
            matrix_basis[frame, id] = np.array(pbone.matrix_basis)
    return matrix_basis

def make_asset(
    asset: Asset,
    extrude_scale: float=0.5,
    connect_tail_to_unique_child: bool=False,
    extrude_from_parent: bool=False,
    group_per_vertex: int=-1,
    add_root: bool=False,
    do_not_normalize: bool=False,
    collection_name: str='new_collection',
    use_face: bool=True,
):
    """
    Args:
    
        extrude_scale: float=0.5, if there is no tails in asset, first calculate the average length between parents and sons, then the length of leaf bone is l*extrude_scale. Otherwise do not affect final results.
        
        connect_tail_to_unique_child: bool=False, if True, the tail of a bone with only one child will be exactly at the head of its child.
        
        extrude_from_parent: bool=False, if True, the orientation of the leaf bone will be the same as its parent.
        
        group_per_vertex: int=-1, number of the largest weights to keep for each vertex. -1 means keep all.
        
        add_root: bool=False, if True, add a root bone at (0, 0, 0).
        
        do_not_normalize: bool=False, if True, do not normalize the skinning weights.
        
        collection_name: str='new_collection', name of the new collection to store objects.
        
        use_face: bool=True, if False, do not export faces.
    """
    
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    
    # 1. if there are meshes, make meshes
    
    objects = []
    mesh_names = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            objects.append(v)
            mesh_names.append(v.name)
    
    if len(objects) == 0:
        mesh_names = [f"mesh_{i}" for i in range(asset.P)]
    if len(objects)==0 and asset.vertices is not None:
        
        if asset.mesh_names is not None:
            mesh_names = asset.mesh_names
        
        for i in range(asset.P):
            mesh = bpy.data.meshes.new(f"data_{mesh_names[i]}")
            v = asset.vertices[asset.get_vertex_slice(i)]
            if not use_face or (asset.faces is None or asset.face_bias is None or asset.vertex_bias is None):
                mesh.from_pydata(v, [], [])
            else:
                if i == 0:
                    mesh.from_pydata(v, [], asset.faces[asset.get_face_slice(i)])
                else:
                    mesh.from_pydata(v, [], asset.faces[asset.get_face_slice(i)]-asset.vertex_bias[i-1])
            mesh.update()
            
            # make object from mesh
            object = bpy.data.objects.new(mesh_names[i], mesh)
            objects.append(object)
            
            # add object to scene collection
            collection.objects.link(object)
    
    # 2. if there is armature, process tails and make armature
    if len(bpy.data.armatures) > 0:
        armature = bpy.data.armatures[0]
        armature_name = armature.name
        joint_names = [b.name for b in armature.bones]
    else:
        armature = None
        armature_name = 'Armature'
        joint_names = asset.joint_names if asset.joint_names is not None else [f"bone_{i}" for i in range(asset.J)]
    
    if armature is None and asset.joints is not None and asset.parents is not None:
        joints = asset.joints
        if asset.tails is None:
            tails = joints.copy()
            connect_tail_to_unique_child = True
            extrude_from_parent = True
        else:
            tails = asset.tails
        
        root_tail = False
        root_id = asset.root
        
        length_sum = 0.
        sons = defaultdict(list)
        for i in range(len(asset.parents)):
            p = asset.parents[i]
            if p == -1:
                continue
            sons[p].append(i)
            length_sum += np.linalg.norm(joints[i] - joints[p])
        if asset.J <= 1:
            length = 1.0
        else:
            length_avg = length_sum / max(len(asset.parents) - 1, 1)
            length = length_avg * extrude_scale
        
        for i in range(len(asset.parents)):
            p = asset.parents[i]
            if p == -1:
                continue
            sons[p].append(i)
            d = np.linalg.norm(joints[i] - joints[p])
            if d <= length * 1e-2:
                max_d = max(length, 1e-5)
                joints[i] += np.random.randn(3) * max_d * 1e-2
        if connect_tail_to_unique_child:
            for i in range(len(asset.parents)):
                if len(sons[i]) == 1:
                    child = sons[i][0]
                    tails[i] = joints[child]
                    if root_id == i:
                        root_tail = True
        
        if extrude_from_parent:
            for i in range(len(asset.parents)):
                if len(sons[i]) != 1 and asset.parents[i] != -1:
                    p = asset.parents[i]
                    d = joints[i] - joints[p]
                    if np.linalg.norm(d) < 1e-6:
                        d = np.array([0., 0., 1.]) # in case son.head == parent.head
                    else:
                        d = d / np.linalg.norm(d)
                    tails[i] = joints[i] + d * length
        if root_tail is False:
            tails[root_id] = joints[root_id] + np.array([0., 0., length])
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.data.armatures.get('Armature')
        armature_name = asset.armature_name if asset.armature_name is not None else 'Armature'
        
        edit_bones = armature.edit_bones
        
        if add_root:
            bone_root = edit_bones.get('Bone')
            root_name = 'Root'
            x = 0
            while root_name in joint_names:
                root_name = f'Root_{x}'
                x += 1
            bone_root.name = root_name
            bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
        else:
            bone_root = edit_bones.get('Bone')
            bone_root.name = joint_names[0]
            bone_root.head = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
            bone_root.tail = Vector((tails[0, 0], tails[0, 1], tails[0, 2]))
        
        def extrude_bone(
            edit_bones,
            name: str,
            parent_name: str,
            head: Tuple[float, float, float],
            tail: Tuple[float, float, float],
        ):
            bone = edit_bones.new(name)
            bone.head = Vector((head[0], head[1], head[2]))
            bone.tail = Vector((tail[0], tail[1], tail[2]))
            bone.name = name
            parent_bone = edit_bones.get(parent_name)
            bone.parent = parent_bone
            bone.use_connect = False
            assert not np.isnan(head).any(), f"nan found in head of bone {name}"
            assert not np.isnan(tail).any(), f"nan found in tail of bone {name}"
        
        for u in asset.dfs_order:
            if add_root is False and u==0:
                continue
            pname = joint_names[u] if asset.parents[u] == -1 else joint_names[asset.parents[u]]
            extrude_bone(edit_bones, joint_names[u], pname, joints[u], tails[u])
            
        is_mixamo = any("mixamorig:" in name for name in joint_names)
        if is_mixamo:
            for bone in edit_bones:
                bone.select = True
            bpy.ops.armature.calculate_roll(type='GLOBAL_NEG_Y')
            
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 3. if there is skin, set vertex groups
    if asset.skin is not None and armature is not None and len(objects) > 0:
        # must set to object mode to enable parent_set
        bpy.ops.object.mode_set(mode='OBJECT')
        N = len(objects)
        objects = bpy.data.objects
        for o in bpy.context.selected_objects:
            o.select_set(False)
        for i in range(N):
            skin = asset.skin[asset.get_vertex_slice(i)]
            ob = objects[mesh_names[i]]
            armature_b = bpy.data.objects[armature_name]
            ob.select_set(True)
            armature_b.select_set(True)
            bpy.ops.object.parent_set(type='ARMATURE_NAME')
            # sparsify
            argsorted = np.argsort(-skin, axis=1)
            vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]
            group_per_vertex = min(group_per_vertex, skin.shape[1])
            if group_per_vertex == -1:
                group_per_vertex = vertex_group_reweight.shape[-1]
            if not do_not_normalize:
                vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[...,None]
            # clean vertex groups first in case skin exists
            for name in joint_names:
                ob.vertex_groups[name].remove(range(990))
            for v, w in enumerate(skin):
                for ii in range(group_per_vertex):
                    j = argsorted[v, ii]
                    n = joint_names[j]
                    ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')
    
    def to_matrix(x: ndarray):
        return Matrix((x[0, :], x[1, :], x[2, :], x[3, :]))
    
    if asset.matrix_world is None:
        matrix_world = to_matrix(np.eye(4))
    else:
        matrix_world = to_matrix(asset.matrix_world)
    if armature is not None:
        bpy.data.objects[armature_name].matrix_world = matrix_world
    
    # 4. if there is animation, set keyframes
    if asset.matrix_basis is not None and asset.matrix_local is not None and armature is not None:
        matrix_basis = asset.matrix_basis
        matrix_local = asset.matrix_local
        objects = bpy.data.objects
        for o in bpy.context.selected_objects:
            o.select_set(False)
        armature = bpy.data.objects[armature_name]
        armature.select_set(True)
        armature.matrix_world = matrix_world
        frames = matrix_basis.shape[0]
        
        # change matrix_local
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        for (id, name) in enumerate(joint_names):
            # matrix_local of pose bone
            bpy.context.active_object.data.edit_bones[id].matrix = to_matrix(matrix_local[id])
        bpy.ops.object.mode_set(mode='OBJECT')
        for (id, name) in enumerate(joint_names):
            pbone = armature.pose.bones.get(name)
            for frame in range(frames):
                bpy.context.scene.frame_set(frame + 1)
                q = to_matrix(matrix_basis[frame, id])
                if pbone.rotation_mode == "QUATERNION":
                    pbone.rotation_quaternion = q.to_quaternion()
                    pbone.keyframe_insert(data_path = 'rotation_quaternion')
                else:
                    pbone.rotation_euler = q.to_euler()
                    pbone.keyframe_insert(data_path = 'rotation_euler')
                pbone.location = q.to_translation()
                pbone.keyframe_insert(data_path = 'location')
        bpy.ops.object.mode_set(mode='OBJECT')

def _umeyama_similarity(src: ndarray, tgt: ndarray) -> ndarray:
    assert src.shape == tgt.shape
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    
    # cross-covariance
    C = (src_c.T @ tgt_c) / n
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_src = (src_c ** 2).sum() / n
    scale = S.sum() / var_src
    t = tgt_mean - scale * R @ src_mean
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T

def _pca_similarity(
    src: ndarray,
    tgt: ndarray,
    max_points: int=4096,
) -> ndarray:
    if src.shape[0] > max_points:
        src = src[np.random.choice(src.shape[0], max_points, replace=False)]
    if tgt.shape[0] > max_points:
        tgt = tgt[np.random.choice(tgt.shape[0], max_points, replace=False)]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    U_src, _, _ = np.linalg.svd(src_c.T @ src_c)
    U_tgt, _, _ = np.linalg.svd(tgt_c.T @ tgt_c)
    R = U_tgt @ U_src.T
    if np.linalg.det(R) < 0:
        U_tgt[:, -1] *= -1
        R = U_tgt @ U_src.T
    scale = np.sqrt((tgt_c ** 2).sum() / (src_c ** 2).sum())
    t = tgt_mean - scale * R @ src_mean
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T

def estimate_similarity_transform(
    src: ndarray,
    tgt: ndarray,
    max_points: int=4096,
) -> ndarray:
    """
    src: (N, 3)
    tgt: (M, 3)
    return: (4, 4) similarity transform matrix
    """
    if src.shape[0] == tgt.shape[0]:
        return _umeyama_similarity(src, tgt)
    return _pca_similarity(src, tgt, max_points)

def transfer_rigging(
    source_asset: Asset,
    target_path: str,
    export_path: str,
    **kwargs,
):
    assert source_asset.matrix_local is not None
    assert source_asset.parents is not None
    
    target_asset = BpyParser.load(filepath=target_path)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    data_types = [
        bpy.data.actions,
        bpy.data.armatures,
    ]
    for data_collection in data_types:
        for item in data_collection:
            data_collection.remove(item)
    
    source_vertices = source_asset.vertices # (n, 3)
    target_vertices = target_asset.vertices # (m, 3)
    assert source_vertices is not None and target_vertices is not None
    target_asset.matrix_local = source_asset.matrix_local.copy()
    target_asset.matrix_basis = source_asset.matrix_basis.copy() if source_asset.matrix_basis is not None else None
    
    source_joints = source_asset.joints
    assert source_joints is not None
    
    max_points = kwargs.pop('max_points', 4096) if kwargs.get('max_points') is not None else 4096
    T = estimate_similarity_transform(src=source_vertices, tgt=target_vertices, max_points=max_points)
    source_joints_h = np.concatenate([
        source_joints, np.ones((len(source_joints), 1))
    ], axis=1)
    target_joints = (T @ source_joints_h.T).T[:, :3]
    target_asset.matrix_local[:, :3, 3] = target_joints
    target_asset.parents = source_asset.parents.copy()
    target_asset.lengths = source_asset.lengths.copy() if source_asset.lengths is not None else None
    target_asset.joint_names = source_asset.joint_names.copy() if source_asset.joint_names is not None else None
    
    if source_asset.skin is not None:
        from scipy.spatial import cKDTree
        source_skin = source_asset.skin # (n, J)
        
        source_vertices_h = np.concatenate([
            source_vertices, np.ones((len(source_vertices), 1))
        ], axis=1)
        source_vertices = (T @ source_vertices_h.T).T[:, :3]
        tree = cKDTree(source_vertices)
        dists, idx = tree.query(target_vertices, k=1)
        target_asset.skin = source_skin[idx]
    
    BpyParser.export(target_asset, export_path, use_origin=True, **kwargs)
    clean_bpy()