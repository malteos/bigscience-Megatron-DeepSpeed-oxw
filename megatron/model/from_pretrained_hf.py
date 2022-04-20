import re
import torch

from transformers.models.gpt2 import GPT2Model, GPT2Config, GPT2LMHeadModel

from megatron import print_rank_0

MODULE_PREFIX = None
# MODULE_PREFIX = r'module\.'


HF_STATE_DICT_MAPPINGS = {
    r'tied_modules\.embed\.word_embeddings\.weight': {
        'hf_k': 'transformer.wte.weight',
        'vocab_offset': True,
    },
    r'tied_modules\.embed\.position_embeddings\.weight': {
        'hf_k': 'transformer.wpe.weight',
    },
    r'([0-9]+)\.input_layernorm\.weight': {
        'hf_k': 'transformer.h.<LAYER>.ln_1.weight'
    },
    r'([0-9]+)\.input_layernorm\.bias': {
        'hf_k': 'transformer.h.<LAYER>.ln_1.bias'
    },
    r'([0-9]+)\.self_attention\.query_key_value\.weight': {
        'hf_k': 'transformer.h.<LAYER>.attn.c_attn.weight',
        'transpose': True,
    },
    r'([0-9]+)\.self_attention\.query_key_value\.bias': {
        'hf_k': 'transformer.h.<LAYER>.attn.c_attn.bias',
    },
    r'([0-9]+)\.self_attention\.dense\.weight': {
        'hf_k': 'transformer.h.<LAYER>.attn.c_proj.weight',
    },
    r'([0-9]+)\.self_attention\.dense\.bias': {
        'hf_k': 'transformer.h.<LAYER>.attn.c_proj.bias',
    },
    r'([0-9]+)\.post_attention_layernorm\.weight': {
        'hf_k': 'transformer.h.<LAYER>.ln_2.weight',
    },
    r'([0-9]+)\.post_attention_layernorm\.bias': {
        'hf_k': 'transformer.h.<LAYER>.ln_2.bias',
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.weight': {
        'hf_k': 'transformer.h.<LAYER>.mlp.c_fc.weight',
        'transpose': True,
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.bias': {
        'hf_k': 'transformer.h.<LAYER>.mlp.c_fc.bias',
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.weight': {
        'hf_k': 'transformer.h.<LAYER>.mlp.c_proj.weight',
        'transpose': True,
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.bias': {
        'hf_k': 'transformer.h.<LAYER>.mlp.c_proj.bias',
    },
    r'([0-9]+)\.bias': {
        'hf_k': 'transformer.ln_f.bias'
    },
    r'([0-9]+)\.weight': {
        'hf_k': 'transformer.ln_f.weight'
    },
}


def get_state_dict_from_hf(input_state_dict, hf_model_name_or_path: str, fp16: float):
    print_rank_0(f'## Loading Huggingface model: {hf_model_name_or_path}')

    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name_or_path)

    if fp16:
        print_rank_0(f'## Converting HF model to fp16')

        hf_model = hf_model.half()

    hf_vocab_size = len(hf_model.transformer.wte.weight)
    hf_sd = hf_model.state_dict()

    layer_offset = 3  # depends on learned pos embeddings
    matched_keys = set()

    print_rank_0(f'## Inputs state dict keys: {input_state_dict.keys()}')

    for k in input_state_dict.keys():

        for mapping_pattern, _mapping in HF_STATE_DICT_MAPPINGS.items():
            if MODULE_PREFIX:
                mapping_pattern = MODULE_PREFIX + mapping_pattern

            match = re.search(mapping_pattern, k)

            if match:
                hf_mapping = _mapping
                hf_k = hf_mapping['hf_k']

                if match.groups():
                    idx = int(match.group(1))
                    hf_idx = idx - layer_offset
                    hf_k = hf_k.replace('<LAYER>', str(hf_idx))

                # check if values match
                original_v = input_state_dict[k]
                hf_v = hf_sd[hf_k]

                if 'transpose' in hf_mapping and hf_mapping['transpose']:
                    hf_v = hf_v.t()

                if 'vocab_offset' in hf_mapping and hf_mapping['vocab_offset']:
                    # concat remaining from orignal value
                    hf_v = torch.cat((hf_v, original_v[hf_vocab_size:, :]))
                    # print('new shape', hf_v.shape)

                if original_v.shape != hf_v.shape:
                    raise ValueError(f'Shapes do not match: {k} = {original_v.shape}; {hf_k} = {hf_v.shape}')

                input_state_dict[k] = hf_v
                matched_keys.add(k)

    # Check if all keys were matched
    not_matched_keys = set(input_state_dict.keys()) - matched_keys

    if len(not_matched_keys) > 0:
        raise ValueError('Not matched keys: %s' % not_matched_keys)

    print_rank_0(f'## Matched state dict keys: {len(matched_keys)}')

    return input_state_dict
