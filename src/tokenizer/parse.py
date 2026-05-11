from copy import deepcopy

from .spec import Tokenizer
from .tokenizer_part import TokenizerPart

def get_tokenizer(**kwargs) -> Tokenizer:
    __target__ = kwargs.get('__target__')
    assert __target__ is not None, "do not find `__target__` in tokenizer config"
    del kwargs['__target__']
    MAP = {
        'tokenizer_part': TokenizerPart,
    }
    return MAP[__target__].parse(**deepcopy(kwargs))
