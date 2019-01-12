__version__ = '1.0.0'

try:
    __ETW_PT_UTILS_SETUP__
except:
    __ETW_PT_UTILS_SETUP__ = False

if not __ETW_PT_UTILS_SETUP__:
    from .pytorch_utils import *
    from .persistent_dataloader import DataLoader
    from .viz import *
    from .seq import Seq
