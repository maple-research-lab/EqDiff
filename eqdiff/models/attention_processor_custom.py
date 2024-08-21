from importlib import import_module
from typing import Callable, Optional, Union
import math

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from diffusers.models.attention_processor import Attention

from einops import rearrange, repeat

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

res_max = 16

class DepthWiseSeperableConv(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        if 'groups' in kwargs:
            # ignoring groups for Depthwise Sep Conv
            del kwargs['groups']
        
        self.depthwise = nn.Conv2d(in_dim, in_dim, *args, groups=in_dim, **kwargs)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class CustomDiffusionAttnProcessor(nn.Module):
    r"""
    Processor for implementing attention for the Custom Diffusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv=True,
        only_train_k=False,
        only_train_v=False,
        train_q_out=True,
        train_res=False,
        use_learnable_ref_mask=False,
        use_integra_ref_mask=False,
        hidden_size=None,
        cross_attention_dim=None,
        out_bias=True,
        dropout=0.0,
        mask_hook=None,
        place_in_unet=None,
        controller=None,
        controller_res=None
    ):
        super().__init__()
        self.attention_op = None
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.train_res = train_res
        self.use_learnable_ref_mask = use_learnable_ref_mask
        self.use_integra_ref_mask = use_integra_ref_mask

        self.mask_hook = mask_hook
        self.place_in_unet = place_in_unet

        self.only_train_k = only_train_k
        self.only_train_v = only_train_v

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.controller = controller
        self.controller_res = controller_res
        # if self.controller_res is not None:
        #     print(f"self.controller_res: {self.controller_res}")

        if self.train_res:
            self.to_out_res = nn.ModuleList([])
            self.to_k_res = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_res = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            if self.use_integra_ref_mask:
                self.to_out_res.append(nn.Linear(hidden_size*2, hidden_size+1, bias=out_bias))
                self.to_out_res.append(nn.Dropout(dropout))
            else:
                self.to_out_res.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
                self.to_out_res.append(nn.Dropout(dropout))
            nn.init.zeros_(self.to_out_res[0].weight)
            nn.init.zeros_(self.to_out_res[0].bias)
            if self.use_learnable_ref_mask:
                self.mask_res = nn.ModuleList()
                self.mask_res.append(nn.Conv2d(hidden_size*2, hidden_size, kernel_size=3, padding=1))
                self.mask_res.append(nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1))
                self.mask_res = nn.Sequential(*self.mask_res)
                for conv in self.mask_res:
                    nn.init.zeros_(conv.weight)
                    nn.init.zeros_(conv.bias)

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, res_states=None, 
                 attention_mask=None, save_attention_map=True, ref_input_mask=None, ref_output_mask=None):
        batch_size, sequence_length, area_h = hidden_states.shape
        area_eh = encoder_hidden_states.shape[-1] if encoder_hidden_states is not None else None

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))
        
        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.only_train_k or self.only_train_v:
            if self.only_train_k:
                key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
                value = attn.to_v(encoder_hidden_states)
                key = key.to(attn.to_q.weight.dtype)
                value = value.to(attn.to_q.weight.dtype)
                # print(f'only train k')
            elif self.only_train_v:
                key = attn.to_k(encoder_hidden_states)
                value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
                device = key.device
                if hasattr(self, 'index'):
                    if self.index is not None:
                        def batch_index_to_one_hot(batch_tensor, sequence_length):
                            batch_size = batch_tensor.size(0)
                            one_hot_tensor = torch.zeros(size=(batch_size, sequence_length, 1), device=device)
                            for i, index in enumerate(batch_tensor):
                                if not index == 0:#not found any rare token
                                    one_hot_tensor[i, index] = 1
                            return one_hot_tensor
                        value_ori = attn.to_v(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
                        index_mask = batch_index_to_one_hot(self.index, value_ori.shape[1])
                        print(f'self.index:{self.index}, index_mask:{index_mask}')
                        bs = key.shape[0]
                        print(f'bs:{bs}, value:{value.shape}, value_ori:{value_ori.shape}, index_mask:{index_mask.shape}')
                        value = value * index_mask + value_ori * (1-index_mask)
                
                key = key.to(attn.to_q.weight.dtype)
                value = value.to(attn.to_q.weight.dtype)
                # print(f'only train v')
            else:
                raise ValueError
        else:
            if self.train_kv:
                key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
                value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
                key = key.to(attn.to_q.weight.dtype)
                value = value.to(attn.to_q.weight.dtype)
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        # query: torch.Size([4, 4096, 320]), key: torch.Size([4, 4096, 320]), value: torch.Size([4, 4096, 320])
        if query.shape[-2] <= res_max**2 or not is_xformers_available():
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            # attention map extraction
            # since reference encoder self-attn also use itself as encoder_hidden_states, so
            # is_cross should be set by res_states (mid, up) and crossattn (down)
            #is_cross = crossattn and (res_states is None)
            is_cross = crossattn and not (area_h == area_eh)
            if self.controller is not None and save_attention_map:
                #print(f'controller is_cross:{is_cross}, crossattn:{crossattn}, res_states:{res_states is None}, {attention_probs.shape}, ')
                self.controller(attention_probs.clone(), is_cross, self.place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            # Using XFormers
            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.batch_to_head_dim(hidden_states)  

        if self.use_integra_ref_mask:
            hidden_states_ori = hidden_states

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        # res self-attention
        if res_states is not None and self.train_res:
            key_res = self.to_k_res(res_states.to(self.to_k_res.weight.dtype))
            value_res = self.to_v_res(res_states.to(self.to_v_res.weight.dtype))
            key_res = key_res.to(attn.to_q.weight.dtype)
            value_res = value_res.to(attn.to_q.weight.dtype)

            if ref_input_mask is not None:
                print(f'key_res:{key_res.shape}, ref_input_mask:{ref_input_mask.shape}')
                l = key_res.shape[1]
                b, _, h, w = ref_input_mask.shape
                ref_input_mask = F.interpolate(ref_input_mask, size=(int(math.sqrt(l)), int(math.sqrt(l))), mode='bilinear')
                ref_input_mask = ref_input_mask.reshape(b, 1, -1).permute(0,2,1).contiguous()
                key_res = key_res + torch.log(ref_input_mask + 1e-6)

            key_res = attn.head_to_batch_dim(key_res)
            value_res = attn.head_to_batch_dim(value_res)
            if query.shape[-2] <= res_max**2 or not is_xformers_available():
                attention_probs_res = attn.get_attention_scores(query, key_res, attention_mask)
                if self.controller_res is not None and save_attention_map:
                    self.controller_res(attention_probs_res.clone(), False, self.place_in_unet)
                hidden_states_res = torch.bmm(attention_probs_res, value_res)
            else:
                hidden_states_res = xformers.ops.memory_efficient_attention(query, key_res, value_res, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)

            hidden_states_res = attn.batch_to_head_dim(hidden_states_res)

            if self.use_integra_ref_mask:
                _, _, c = hidden_states_res.shape
                out = self.to_out_res[0](torch.cat([hidden_states_res, hidden_states_ori], dim=-1))
                hidden_states_res, mask = out[:,:,:c], torch.clamp(out[:,:,c:], min=0., max=1.)
                # from torchvision.utils import save_image
                # from datetime import datetime
                # import os
                # b,l,c = mask.shape
                # h = int(math.sqrt(l))
                # filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
                # os.makedirs('./tmp/mask', exist_ok=True)
                # save_image(mask.detach().reshape(b, h, h, c).permute(0,3,1,2), os.path.join('./tmp/mask', filename))
                # print(f'mask:{mask.min()},{mask.max()}')

                assert self.mask_hook is not None
                b,l,c = mask.shape
                h = int(math.sqrt(l))
                self.mask_hook(mask.detach().reshape(b, h, h, c).permute(0,3,1,2), place_in_unet=self.place_in_unet)
                # for k,v in self.mask_hook.step_store.items():
                #     print(f'attention cal:{k}:{len(v)}')
            else:
                hidden_states_res = self.to_out_res[0](hidden_states_res)
            # print(f'performing res sa calculation:{self.to_out_res[0].weight.sum()} vs {attn.to_out[0].weight.sum()}')
            hidden_states_res = self.to_out_res[1](hidden_states_res)

            assert not (self.use_learnable_ref_mask and ref_output_mask is not None)
            
            if ref_output_mask is not None:
                print(f'ref_out_mask:{ref_output_mask.shape}, hidden_states_res:{hidden_states_res.shape}')
                hidden_states_res = hidden_states_res * ref_output_mask
            
            if self.use_learnable_ref_mask:
                print(f'hidden_states:{hidden_states.shape}, hidden_states_res:{hidden_states_res.shape}')
                l = hidden_states.shape[1]
                h, w = int(math.sqrt(l)), int(math.sqrt(l))
                mask_conv_input = torch.cat([hidden_states, hidden_states_res], -1)#(b, hw, 2*c)
                mask_conv_input = rearrange(mask_conv_input, 'b (h w) c -> b c h w', h=h, w=w)
                mask = F.sigmoid(self.mask_res(mask_conv_input))
                mask = rearrange(mask, 'b c h w -> b (h w) c')
                hidden_states = hidden_states*(1-mask) + hidden_states_res*mask
                return hidden_states

            if self.use_integra_ref_mask:
                hidden_states = hidden_states*(1-mask) + hidden_states_res*mask
            else:
                hidden_states = hidden_states + hidden_states_res


        return hidden_states


class CustomDiffusionXFormersAttnProcessor(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.

    Args:
    train_kv (`bool`, defaults to `True`):
        Whether to newly train the key and value matrices corresponding to the text features.
    train_q_out (`bool`, defaults to `True`):
        Whether to newly train query matrices corresponding to the latent image features.
    hidden_size (`int`, *optional*, defaults to `None`):
        The hidden size of the attention layer.
    cross_attention_dim (`int`, *optional*, defaults to `None`):
        The number of channels in the `encoder_hidden_states`.
    out_bias (`bool`, defaults to `True`):
        Whether to include the bias parameter in `train_q_out`.
    dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
    attention_op (`Callable`, *optional*, defaults to `None`):
        The base
        [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to use
        as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best operator.
    """

    def __init__(
        self,
        train_kv=True,
        train_q_out=False,
        train_res=False,
        hidden_size=None,
        cross_attention_dim=None,
        out_bias=True,
        dropout=0.0,
        attention_op: Optional[Callable] = None,
        place_in_unet=None,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.train_res = train_res

        self.place_in_unet = place_in_unet

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.attention_op = attention_op

        if self.train_res:
            self.to_out_res = nn.ModuleList([])
            self.to_k_res = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_res = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_out_res.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_res.append(nn.Dropout(dropout))
            nn.init.zeros_(self.to_out_res[0].weight)
            nn.init.zeros_(self.to_out_res[0].bias)

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, res_states=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

         # res self-attention
        if res_states is not None and self.train_res:
            key_res = self.to_k_res(res_states.to(self.to_k_res.weight.dtype))
            value_res = self.to_v_res(res_states.to(self.to_v_res.weight.dtype))
            key_res = key_res.to(attn.to_q.weight.dtype)
            value_res = value_res.to(attn.to_q.weight.dtype)
            key_res = attn.head_to_batch_dim(key_res)
            value_res = attn.head_to_batch_dim(value_res)

            hidden_states_res = xformers.ops.memory_efficient_attention(query, key_res, value_res, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
            hidden_states_res = hidden_states_res.to(query.dtype)

            hidden_states_res = attn.batch_to_head_dim(hidden_states_res)
            hidden_states_res = self.to_out_res[0](hidden_states_res)
            #print(f'performing res sa calculation:{self.to_out_res[0].weight.sum()}')
            hidden_states_res = self.to_out_res[1](hidden_states_res)

            hidden_states = hidden_states + hidden_states_res

        return hidden_states
