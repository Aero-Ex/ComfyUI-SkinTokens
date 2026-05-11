from .skin_cvae_model import SkinCVAEModel
from .skin_fsq_cvae_model import SkinFSQCVAEModel

def get_model_cvae(
    pretrained_path: str=None,
    **kwargs
) -> SkinCVAEModel:
    model = SkinCVAEModel(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
    return model

def get_model_fsq_cvae(
    pretrained_path: str=None,
    **kwargs
) -> SkinFSQCVAEModel:
    model = SkinFSQCVAEModel(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
    return model