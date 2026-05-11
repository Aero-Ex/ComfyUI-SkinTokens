import os
import folder_paths

# Add custom model folder for SkinTokens
skintoken_models_dir = os.path.join(folder_paths.models_dir, "skintoken")
if not os.path.exists(skintoken_models_dir):
    os.makedirs(skintoken_models_dir)

# Register the skintoken model type
folder_paths.folder_names_and_paths["skintoken"] = ([skintoken_models_dir], folder_paths.supported_pt_extensions)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
