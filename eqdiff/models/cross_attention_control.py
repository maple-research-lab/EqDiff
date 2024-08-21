import torch
import torch.nn.functional as F

from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.models.attention import BasicTransformerBlock
from eqdiff.models.attention import BasicTransformerBlock as _BasicTransformerBlock

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class CrossAttentionControl():

    def __init__(self,
                 unet,
                 fusion_blocks="full",
                 ) -> None:
        self.unet = unet
        assert fusion_blocks in ["midup", "full"]
        self.fusion_blocks = fusion_blocks
        attn_modules = [module for name, module in unet.attn_processors.items() if name.endswith("attn2.processor") ]
        for m in attn_modules:
            m.index = None

    def update(self, index=None):
        if self.fusion_blocks == "full":
            attn_modules = [module for name, module in self.unet.attn_processors.items() if name.endswith("attn2.processor") ]
        else:
            raise ValueError
        assert len(attn_modules) > 0, f'iterating unet attention modules failed.'
        for m in attn_modules:
            if index is not None:
                m.index = index
                print(f'setting index for {m}')
    
    def clear(self):
        if self.fusion_blocks == "full":
            attn_modules = [module for name, module in self.unet.attn_processors.items() if name.endswith("attn2.processor") ]
        else:
            raise ValueError
        assert len(attn_modules) > 0, f'iterating unet attention modules failed.'
        for m in attn_modules:
            m.index = None
            print(f'clearing index for {m}')