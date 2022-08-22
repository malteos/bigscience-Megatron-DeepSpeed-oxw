
# OPT ############
HF_OPT_STATE_DICT_MAPPINGS = {
    # ds state dict key => HF state dict key + convert operation
    r'tied_modules\.embed\.word_embeddings\.weight': {
        'hf_k': 'model.decoder.embed_tokens.weight',
        'vocab_offset': True,  # TODO?
    },
    r'tied_modules\.embed\.position_embeddings\.weight': {
        'hf_k': 'model.decoder.embed_positions.weight',
    },
    r'([0-9]+)\.input_layernorm\.weight': {
        'hf_k': 'model.decoder.layers.<LAYER>.self_attn_layer_norm.weight',
    },
    r'([0-9]+)\.input_layernorm\.bias': {
        'hf_k': 'model.decoder.layers.<LAYER>.self_attn_layer_norm.bias',
    },
    r'([0-9]+)\.self_attention\.query_key_value\.weight': {
        'hf_keys': [
            'model.decoder.layers.<LAYER>.self_attn.q_proj.weight',
            'model.decoder.layers.<LAYER>.self_attn.k_proj.weight',
            'model.decoder.layers.<LAYER>.self_attn.v_proj.weight',
        ],
    },
    r'([0-9]+)\.self_attention\.query_key_value\.bias': {
        'hf_keys': [
            'model.decoder.layers.<LAYER>.self_attn.q_proj.bias',
            'model.decoder.layers.<LAYER>.self_attn.k_proj.bias',
            'model.decoder.layers.<LAYER>.self_attn.v_proj.bias',
        ],
    },
    r'([0-9]+)\.self_attention\.dense\.weight': {
        'hf_k': 'model.decoder.layers.<LAYER>.self_attn.out_proj.weight',
        'transpose': False,  # TODO needed maybe?
    },
    r'([0-9]+)\.self_attention\.dense\.bias': {
        'hf_k': 'model.decoder.layers.<LAYER>.self_attn.out_proj.bias',
    },
    r'([0-9]+)\.post_attention_layernorm\.weight': {
        'hf_k': 'model.decoder.layers.<LAYER>.final_layer_norm.weight',
    },
    r'([0-9]+)\.post_attention_layernorm\.bias': {
        'hf_k': 'model.decoder.layers.<LAYER>.final_layer_norm.bias',
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.weight': {
        'hf_k': 'model.decoder.layers.<LAYER>.fc1.weight',
        #'transpose': True,
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.bias': {
        'hf_k': 'model.decoder.layers.<LAYER>.fc1.bias',
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.weight': {
        'hf_k': 'model.decoder.layers.<LAYER>.fc2.weight',
        # 'transpose': True,
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.bias': {
        'hf_k': 'model.decoder.layers.<LAYER>.fc2.bias',
    },
    # Bigs does not have keys for the following HF keys:
    # - "'model.decoder.layers.0.final_layer_norm.weight',  'model.decoder.layers.0.final_layer_norm.bias'"
    # but there should be ""

    # r'([0-9]+)\.bias': {
    #     'hf_k': '', #'transformer.ln_f.bias'
    # },
    # r'([0-9]+)\.weight': {
    #     'hf_k': '', #'transformer.ln_f.weight'
    # },
}