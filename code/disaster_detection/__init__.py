from .model.ernet import ErNET
from .model.squeeze_ernet import Squeeze_ErNET
from .model.squeeze_ernet_redconv import Squeeze_RedConv
from .model.label_smoothing import LabelSmoothingCrossEntropy

__all__ = [
    'ErNET',
    'Squeeze_ErNET',
    'Squeeze_RedConv',
    'LabelSmoothingCrossEntropy'
] 