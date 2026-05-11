from copy import deepcopy
from dataclasses import dataclass

from .michelangelo.get_model import get_encoder as get_encoder_michelangelo
from .michelangelo.get_model import AlignedShapeLatentPerceiver
from .michelangelo.get_model import get_encoder_simplified as get_encoder_michelangelo_encoder
from .michelangelo.get_model import ShapeAsLatentPerceiverEncoder
from .skin_vae.autoencoders.autoencoder_kl_tripo2 import Tripo2Encoder

@dataclass(frozen=True)
class _MAP_MESH_ENCODER:
    michelangelo = AlignedShapeLatentPerceiver
    michelangelo_encoder = ShapeAsLatentPerceiverEncoder
    tripo = Tripo2Encoder

MAP_MESH_ENCODER = _MAP_MESH_ENCODER()


def get_mesh_encoder(**kwargs):
    MAP = {
        'michelangelo': get_encoder_michelangelo,
        'michelangelo_encoder': get_encoder_michelangelo_encoder,
        'tripo': Tripo2Encoder,
    }
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**deepcopy(kwargs))