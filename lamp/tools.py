import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask

import numpy as np

from typing import List, Optional, Tuple, Union

import random

class LearnableEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LearnableEmbeddings, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded

def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    #print(f'inputs_embeds:{inputs_embeds.sum([2])}')
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            if not idx == 0:# prior preservation use init index=0, should skip this 
                lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
                new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]
                new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]
        #print(f'inj_embedding:{inj_embedding.sum([2])}, new_inputs_embeds:{new_inputs_embeds.sum([2])}')

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def fft_filtering(img, filter_type='lowpass', cutoff_freq=10):
    """
    img: Tensor with shape of [b, c, h, w]
    """
    # Convert image to grayscale
    tensor = img.float()
    grayscale_tensor = torch.mean(tensor, dim=1, keepdim=True)
    gray_img = grayscale_tensor.repeat(1, 3, 1, 1)
    

    # Perform Fourier Transform
    f = np.fft.fft2(gray_img.cpu())
    fshift = np.fft.fftshift(f)

    # Define the filter
    batchs, channels, rows, cols = gray_img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)

    if filter_type == 'lowpass':
        mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1
    elif filter_type == 'highpass':
        mask[:crow - cutoff_freq, :] = 1
        mask[crow + cutoff_freq:, :] = 1
        mask[:, :ccol - cutoff_freq] = 1
        mask[:, ccol + cutoff_freq:] = 1

    # Apply the filter
    fshift_filtered = fshift * mask

    # Perform inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).real
    # img_back = np.uint8(img_back)

    return torch.tensor(img_back).to('cuda')

# def generate_random_bool():
#     return random.choice([True, False])

def generate_random_bool(grayprob=0):
    return random.choices([True, False], weights=[grayprob, 1-grayprob], k=1)[0]


def split_fft_values(images, flag_to_combine, high_freq_percentage, low_freq_percentage):
    hfs = fft_filtering(images, filter_type='highpass', cutoff_freq=high_freq_percentage)
    lfs = fft_filtering(images, filter_type='lowpass', cutoff_freq=low_freq_percentage)
    target_images = []
    for bz, (flag, hf, lf) in enumerate(zip(flag_to_combine, hfs, lfs)):
        if flag == 0:#lf
            target_images.append(lf)
        elif flag == 1:#hf
            target_images.append(hf)
        elif flag == 2:#lf+hf
            target_images.append(images[bz])
        else:
            raise ValueError(f'flag is not right')
    target_images = torch.stack(target_images, 0)
    return torch.cat([hfs, lfs], 0), target_images

def combine_embeddings(freq_embeddings, flag_to_combine):
    # freq_embeddings:(b*2, l, c) hf concat lf
    hf_embeddings, lf_embeddings = freq_embeddings.chunk(2)
    final_freq_embeddings = []
    for bs, (flag, hf_embedding, lf_embedding) in enumerate(zip(flag_to_combine, hf_embeddings, lf_embeddings)):
        if flag == 0:#lf
            final_freq_embeddings.append(lf_embedding)
        elif flag == 1:#hf
            final_freq_embeddings.append(hf_embedding)
        elif flag == 2: #hf+lf
            final_freq_embeddings.append(lf_embedding+hf_embedding)
        else:
            raise ValueError(f'the flag is not appropriate in the flag_to_combine')
    return torch.stack(final_freq_embeddings, 0)#(b, l, c)