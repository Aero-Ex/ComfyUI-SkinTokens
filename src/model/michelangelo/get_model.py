import torch

from .models.tsal.sal_perceiver import AlignedShapeLatentPerceiver, ShapeAsLatentPerceiverEncoder

def get_encoder(
    pretrained_path: str=None,
    freeze_decoder: bool=False,
    **kwargs
) -> AlignedShapeLatentPerceiver:
    model = AlignedShapeLatentPerceiver(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
    if freeze_decoder:
        model.geo_decoder.requires_grad_(False)
        model.encoder.query.requires_grad_(False)
        model.pre_kl.requires_grad_(False)
        model.post_kl.requires_grad_(False)
        model.transformer.requires_grad_(False)
    return model

def get_encoder_simplified(
    pretrained_path: str=None,
    **kwargs
) -> ShapeAsLatentPerceiverEncoder:
    model = ShapeAsLatentPerceiverEncoder(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
    return model