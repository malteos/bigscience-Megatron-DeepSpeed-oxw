import re

import collections

import torch


from megatron import print_rank_0

MODULE_PREFIX = None
# MODULE_PREFIX = r'module\.'



INPUT_STATE_DICT_MAPPINGS = {
    r'tied_modules\.embed\.word_embeddings\.weight': {
        'input_key': 'model.language_model.embedding.word_embeddings.weight',
        #'vocab_offset': True,
    },
    r'tied_modules\.embed\.position_embeddings\.weight': {
        'input_key': 'model.language_model.embedding.position_embeddings.weight',
    },
    r'([0-9]+)\.input_layernorm\.weight': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.input_layernorm.weight'
    },
    r'([0-9]+)\.input_layernorm\.bias': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.input_layernorm.bias'
    },
    r'([0-9]+)\.self_attention\.query_key_value\.weight': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.attention.query_key_value.weight',
        #'transpose': True,
    },
    r'([0-9]+)\.self_attention\.query_key_value\.bias': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.attention.query_key_value.bias',
    },
    r'([0-9]+)\.self_attention\.dense\.weight': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.attention.dense.weight',
    },
    r'([0-9]+)\.self_attention\.dense\.bias': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.attention.dense.bias',
    },
    r'([0-9]+)\.post_attention_layernorm\.weight': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.post_attention_layernorm.weight',
    },
    r'([0-9]+)\.post_attention_layernorm\.bias': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.post_attention_layernorm.bias',
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.weight': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.mlp.dense_h_to_4h.weight',
        #'transpose': True,
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.bias': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.mlp.dense_h_to_4h.bias',
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.weight': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.mlp.dense_4h_to_h.weight',
        #'transpose': True,
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.bias': {
        'input_key': 'model.language_model.transformer.layers.<LAYER>.mlp.dense_4h_to_h.bias',
    },
    r'([0-9]+)\.bias': {
        'input_key': 'model.language_model.transformer.final_layernorm.bias'
    },
    r'([0-9]+)\.weight': {
        'input_key': 'model.language_model.transformer.final_layernorm.weight'
    },
}


def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_state_dict_from_meg(update_state_dict, input_checkpoint_path: str):
    print_rank_0(f'## Loading Meg model: {input_checkpoint_path}')

    meg_state_dict = torch.load(input_checkpoint_path, map_location="cpu")

    input_state_dict = flatten(meg_state_dict, sep='.')
    input_vocab_size = len(input_state_dict['model.language_model.embedding.word_embeddings.weight'])

    layer_offset = 3  # depends on learned pos embeddings
    matched_keys = set()

    print_rank_0(f'## Update state dict keys: {update_state_dict.keys()}')

    for k in update_state_dict.keys():

        for mapping_pattern, _mapping in INPUT_STATE_DICT_MAPPINGS.items():
            if MODULE_PREFIX:
                mapping_pattern = MODULE_PREFIX + mapping_pattern

            match = re.search(mapping_pattern, k)

            if match:
                hf_mapping = _mapping
                input_key = hf_mapping['input_key']

                if match.groups():
                    idx = int(match.group(1))
                    hf_idx = idx - layer_offset
                    input_key = input_key.replace('<LAYER>', str(hf_idx))

                # check if values match
                original_v = update_state_dict[k]
                input_v = input_state_dict[input_key]

                if 'transpose' in hf_mapping and hf_mapping['transpose']:
                    input_v = input_v.t()

                if 'vocab_offset' in hf_mapping and hf_mapping['vocab_offset']:
                    # concat remaining from orignal value
                    input_v = torch.cat((input_v, original_v[input_vocab_size:, :]))
                    # print('new shape', hf_v.shape)

                if original_v.shape != input_v.shape:
                    raise ValueError(f'Shapes do not match: {k} = {original_v.shape}; {input_key} = {input_v.shape}')

                update_state_dict[k] = input_v
                matched_keys.add(k)

    # Check if all keys were matched
    not_matched_keys = set(update_state_dict.keys()) - matched_keys

    if len(not_matched_keys) > 0:
        raise ValueError('Not matched keys: %s' % not_matched_keys)

    print_rank_0(f'## Matched state dict keys: {len(matched_keys)}')

    return update_state_dict
