
HF_GPT2_STATE_DICT_MAPPINGS = {
    # ds state dict key => HF state dict key + convert operation
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
        'fix_qkv_ordering_weight': True,
    },
    r'([0-9]+)\.self_attention\.query_key_value\.bias': {
        'hf_k': 'transformer.h.<LAYER>.attn.c_attn.bias',
        'fix_qkv_ordering_bias': True,
    },
    r'([0-9]+)\.self_attention\.dense\.weight': {
        'hf_k': 'transformer.h.<LAYER>.attn.c_proj.weight',
        'transpose': True,
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
