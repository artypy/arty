# check if torch is installed, if yes, from . import cnn, if not, do not import cnn
try:
    import torch
    from ._cnn import CNN
except ImportError:
    pass
