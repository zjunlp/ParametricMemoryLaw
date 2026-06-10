from ..vector_generators.sft.generate_sft_hparams import SFTHyperParams
from ..vector_generators.memft.generate_memft_hparams import MemFTHyperParams
from ..vector_appliers.sft.apply_sft_hparam import ApplySFTHyperParams
from ..vector_appliers.memft.apply_memft_hparam import ApplyMemFTHyperParams
from ..vector_generators.sft.generate_sft import generate_sft
from ..vector_generators.memft.generate_memft import generate_memft
from ..vector_appliers.sft.apply_sft import apply_sft
from ..vector_appliers.memft.apply_memft import apply_memft

import torch

DTYPES_DICT ={
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    'float64': torch.float64,
    "bf16": torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'fp64': torch.float64
}
HYPERPARAMS_CLASS_DICT = {
    'sft':{'train': SFTHyperParams, 'apply': ApplySFTHyperParams},
    'memft':{'train': MemFTHyperParams, 'apply': ApplyMemFTHyperParams},
    
}

METHODS_CLASS_DICT = {
    'sft': {'train': generate_sft, 'apply': apply_sft},
    'memft': {'train': generate_memft, 'apply': apply_memft},
    
}

VLLM_SUPPORTED_METHODS = []
