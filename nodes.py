import os
import subprocess
import sys
import tempfile
import atexit
import time
import requests
from pathlib import Path
from typing import List
from torch import Tensor
import signal
import urllib.parse

import folder_paths

# Add the current directory to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# SkinTokens internal imports
from src.rig_package.info.mixamo_mapper import map_asset_to_mixamo
from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.model.tokenrig import TokenRigResult
from src.tokenizer.parse import get_tokenizer
from src.server.spec import (
    BPY_SERVER,
    get_model,
    object_to_bytes,
    bytes_to_object,
)
from src.data.vertex_group import voxel_skin

# =======================================================================
# LAZY BPY SERVER LOGIC
# =======================================================================

_bpy_server_proc = None
_bpy_server_mode = None  # Tracks if currently running in 'embedded' or 'headless' mode

def start_bpy_server_lazy(headless=False):
    """Starts the Blender python server if it isn't running already."""
    global _bpy_server_proc, _bpy_server_mode
    
    new_mode = "headless" if headless else "embedded"
    
    # If a server is running but it's the WRONG type, kill it so we can restart in the new mode
    if _bpy_server_proc is not None and _bpy_server_proc.poll() is None:
        if _bpy_server_mode != new_mode:
            print(f"[SkinTokens] Restarting server (switching from {_bpy_server_mode} to {new_mode})...")
            cleanup_bpy_server()
        else:
            return  # Already running in the correct mode

    current_dir = os.path.dirname(os.path.abspath(__file__))
    bpy_server_path = os.path.join(current_dir, "bpy_server.py")

    if headless:
        # HEADLESS MODE (System Blender)
        import shutil
        blender_cmd = shutil.which("blender") or "blender"
        args = [blender_cmd, "--background", "--python", bpy_server_path]
        env = os.environ.copy()
        print(f"[SkinTokens] Starting Headless Blender server...")
    else:
        # EMBEDDED MODE (bpy module)
        args = [sys.executable, bpy_server_path]
        # LD_PRELOAD fixes native library conflicts with bpy:
        env = os.environ.copy()
        if os.name != "nt":
            preloads = []
            for lib in ["/usr/lib/libjemalloc.so.2", "/usr/lib/libjpeg.so.8"]:
                if os.path.isfile(lib):
                    preloads.append(lib)
            if preloads:
                existing = env.get("LD_PRELOAD", "")
                env["LD_PRELOAD"] = " ".join(preloads) + (" " + existing if existing else "")
        print(f"[SkinTokens] Starting Embedded bpy server...")

    popen_kwargs = dict(
        args=args,
        cwd=current_dir,
        stdout=None,
        stderr=None,
        env=env,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    _bpy_server_proc = subprocess.Popen(**popen_kwargs)
    _bpy_server_mode = new_mode
    print(f"[SkinTokens] bpy_server.py started (pid={_bpy_server_proc.pid}, mode={_bpy_server_mode})")
    
    wait_for_bpy_server()

def cleanup_bpy_server():
    """Kills the Blender python server on exit."""
    global _bpy_server_proc
    if _bpy_server_proc is not None:
        print(f"[SkinTokens] Terminating bpy_server.py (pid={_bpy_server_proc.pid})")
        try:
            if _bpy_server_proc.poll() is not None:
                return
            if os.name == "nt":
                _bpy_server_proc.terminate()
            else:
                os.killpg(os.getpgid(_bpy_server_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

atexit.register(cleanup_bpy_server)

def wait_for_bpy_server(timeout=30):
    t0 = time.time()
    while True:
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            print("[SkinTokens] bpy_server is ready")
            return
        except Exception:
            if time.time() - t0 > timeout:
                raise RuntimeError("bpy_server failed to start within the timeout period.")
            time.sleep(0.5)

def post_bpy_payload(endpoint: str, payload):
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        request_payload = {"payload_path": payload_path}
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes(request_payload),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


# =======================================================================
# COMFYUI CUSTOM NODES
# =======================================================================

class SkinTokensModelLoader:
    """Loads the SkinTokens model weights."""
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list("skintoken")
        default_model = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
        if default_model not in files:
            files.insert(0, default_model)
            
        return {
            "required": {
                "model_name": (files, {"tooltip": "Select a checkpoint from models/skintoken/. Will auto-download if missing."}),
            }
        }
    
    RETURN_TYPES = ("SKINTOKENS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SkinTokens"
    
    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("skintoken", model_name)
        skintoken_models_dir = os.path.join(folder_paths.models_dir, "skintoken")
        
        if not model_path or not os.path.exists(model_path):
            print(f"[SkinTokens] Model {model_name} not found locally. Attempting to download from HuggingFace...")
            from huggingface_hub import hf_hub_download
            REPO_ID = "VAST-AI/SkinTokens"
            try:
                # Download main model
                hf_hub_download(repo_id=REPO_ID, filename=model_name, local_dir=skintoken_models_dir)
                
                # If it's the default model, also ensure the VAE is downloaded
                if "grpo_1400.ckpt" in model_name:
                    vae_path = "experiments/skin_vae_2_10_32768/last.ckpt"
                    if not os.path.exists(os.path.join(skintoken_models_dir, vae_path)):
                        print(f"[SkinTokens] Downloading required VAE: {vae_path}...")
                        hf_hub_download(repo_id=REPO_ID, filename=vae_path, local_dir=skintoken_models_dir)
                
                model_path = os.path.join(skintoken_models_dir, model_name)
                print(f"[SkinTokens] Successfully downloaded to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model {model_name} from HuggingFace: {e}")
        
        print(f"[SkinTokens] Loading model: {model_path}")
        model = get_model(model_path, hf_path=None)
        assert model.tokenizer_config is not None
        tokenizer = get_tokenizer(**model.tokenizer_config)
        transform = Transform.parse(**model.transform_config["predict_transform"])
        
        state = {
            "model": model,
            "tokenizer": tokenizer,
            "transform": transform,
            "ckpt_path": model_path
        }
        return (state, )


class SkinTokensLoadMesh:
    """Loads a 3D mesh from the ComfyUI input directory or an absolute path."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Absolute or relative path to the 3D model"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    FUNCTION = "load"
    CATEGORY = "SkinTokens"

    def load(self, mesh_path):
        if not mesh_path:
            raise ValueError("No input mesh path provided.")
        
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(folder_paths.get_input_directory(), mesh_path)
            
        if not os.path.exists(mesh_path):
             raise FileNotFoundError(f"Input mesh not found: {mesh_path}")
        return (mesh_path, )


class SkinTokensGenerator:
    """Runs the 3D rig generation using the SkinTokens model and bpy server."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SKINTOKENS_MODEL",),
                "input_mesh": ("STRING", {"forceInput": True, "tooltip": "Connect to a node that outputs a mesh file path"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 20}),
                "use_skeleton": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "use_transfer": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "IMPORTANT: Set to 'Yes' to preserve textures, materials, and original mesh quality from your input file."}),
                "use_postprocess": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "use_mixamo": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "output_format": ([".glb", ".fbx", ".obj"], {"default": ".glb"}),
                "bpy_server_mode": (["Embedded (bpy)", "Headless (Blender)"], {"default": "Embedded (bpy)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_mesh_path",)
    FUNCTION = "generate"
    CATEGORY = "SkinTokens"
    
    def generate(self, model, input_mesh, top_k, top_p, temperature, repetition_penalty, num_beams, 
                 use_skeleton, use_transfer, use_postprocess, use_mixamo, output_format, bpy_server_mode):
        
        if not input_mesh:
            raise ValueError("No input mesh path provided.")
            
        if not os.path.isabs(input_mesh):
            input_path = os.path.join(folder_paths.get_input_directory(), input_mesh)
        else:
            input_path = input_mesh
            
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input mesh not found: {input_path}")
        # Start the blender python server lazily
        headless = (bpy_server_mode == "Headless (Blender)")
        start_bpy_server_lazy(headless=headless)
        
        filepaths = [Path(input_path)]
        
        # Determine output path
        base_name = os.path.splitext(os.path.basename(input_mesh))[0]
        output_dir = folder_paths.get_output_directory()
        output_name = f"{base_name}_rigged_{int(time.time())}{output_format}"
        out_path = Path(output_dir) / output_name
        
        _model = model["model"]
        tokenizer = model["tokenizer"]
        transform = model["transform"]
        
        # Configuration for data loading
        datapath = {
            "data_name": None,
            "loader": "bpy_server",
            "filepaths": {"articulation": [str(p) for p in filepaths]},
        }

        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            datapath=datapath,
        ).split_by_cls()

        module = RigDatasetModule(
            predict_dataset_config=dataset_config,
            predict_transform=transform,
            tokenizer=tokenizer,
            process_fn=_model._process_fn,
        )

        dataloader = module.predict_dataloader()["articulation"]

        for i, batch in enumerate(dataloader):
            # Move to CUDA if it is a tensor
            batch = {
                k: v.to("cuda") if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            if not use_skeleton:
                batch.pop("skeleton_tokens", None)
                batch.pop("skeleton_mask", None)

            batch["generate_kwargs"] = dict(
                max_length=2048,
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                num_return_sequences=1,
                num_beams=int(num_beams),
                do_sample=True,
            )

            if "skeleton_tokens" in batch and "skeleton_mask" in batch:
                mask = batch["skeleton_mask"][0] == 1
                skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
            else:
                skeleton_tokens = None

            # Run inference
            preds: List[TokenRigResult] = _model.predict_step(
                batch,
                skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
                make_asset=True,
            )["results"]

            asset = preds[0].asset
            assert asset is not None

            if use_mixamo:
                asset.joint_names = map_asset_to_mixamo(asset.joints, asset.parents)

            if use_postprocess:
                voxel = asset.voxel(resolution=196)
                asset.skin *= voxel_skin(
                    grid=0,
                    grid_coords=voxel.coords,
                    joints=asset.joints,
                    vertices=asset.vertices,
                    faces=asset.faces,
                    mode="square",
                    voxel_size=voxel.voxel_size,
                )
                asset.normalize_skin()

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if use_transfer:
                payload = dict(
                    source_asset=asset,
                    target_path=asset.path,
                    export_path=str(out_path),
                    group_per_vertex=4,
                )
                res = post_bpy_payload("transfer", payload)
            else:
                payload = dict(
                    asset=asset,
                    filepath=str(out_path),
                    group_per_vertex=4,
                )
                res = post_bpy_payload("export", payload)

            if res != "ok":
                raise RuntimeError(f"bpy_server failed to export: {res}")
            else:
                print(f"[SkinTokens] Successfully exported rigged model to: {out_path}")

        return (str(out_path), )


class SkinTokensRigPreviewer:
    """Passes the mesh path to the frontend Web Extension for 3D visualization and rig manipulation."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SkinTokens"

    def preview(self, mesh_path):
        import urllib.parse
        import os
        import folder_paths
        
        # We need to construct a url for the frontend to fetch the file
        # NOTE: Do NOT prefix with /api here - the JS-side api.apiURL() adds it automatically
        output_dir = folder_paths.get_output_directory()
        print(f"[SkinTokens Previewer] mesh_path={mesh_path}, output_dir={output_dir}")
        try:
            rel_path = os.path.relpath(mesh_path, output_dir)
            if not rel_path.startswith(".."):
                # File is inside output directory
                file_name = os.path.basename(rel_path)
                subfolder = os.path.dirname(rel_path)
                if subfolder == ".":
                    subfolder = ""
                subfolder = subfolder.replace("\\", "/")
                url = f"/view?filename={urllib.parse.quote(file_name)}&type=output&subfolder={urllib.parse.quote(subfolder)}"
            else:
                # File is not in output directory, try input
                input_dir = folder_paths.get_input_directory()
                rel_path_in = os.path.relpath(mesh_path, input_dir)
                if not rel_path_in.startswith(".."):
                    file_name = os.path.basename(rel_path_in)
                    subfolder = os.path.dirname(rel_path_in).replace("\\", "/")
                    if subfolder == ".":
                        subfolder = ""
                    url = f"/view?filename={urllib.parse.quote(file_name)}&type=input&subfolder={urllib.parse.quote(subfolder)}"
                else:
                    url = f"/view?filename={urllib.parse.quote(os.path.basename(mesh_path))}&type=output"
        except:
            url = f"/view?filename={urllib.parse.quote(os.path.basename(mesh_path))}&type=output"
        
        print(f"[SkinTokens Previewer] Generated URL: {url}")
        return {"ui": {"skintokens_mesh": [url]}, "result": (mesh_path,)}

# =======================================================================
# REGISTRATION
# =======================================================================

NODE_CLASS_MAPPINGS = {
    "SkinTokensModelLoader": SkinTokensModelLoader,
    "SkinTokensLoadMesh": SkinTokensLoadMesh,
    "SkinTokensGenerator": SkinTokensGenerator,
    "SkinTokensRigPreviewer": SkinTokensRigPreviewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinTokensModelLoader": "Load SkinTokens Model",
    "SkinTokensLoadMesh": "Load SkinTokens Mesh",
    "SkinTokensGenerator": "SkinTokens Rig Generator",
    "SkinTokensRigPreviewer": "SkinTokens Rig Previewer",
}
