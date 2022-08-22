# Bloom ###########
HF_BLOOM_STATE_DICT_MAPPINGS = {
    # ds state dict key => HF state dict key + convert operation
    r'tied_modules\.embed\.word_embeddings\.weight': {
        'hf_k': 'transformer.word_embeddings.weight',
    },
    r'tied_modules\.embed\.word_embeddings.norm\.weight': {
        'hf_k': 'transformer.word_embeddings_layernorm.weight',
    },
    r'tied_modules\.embed\.word_embeddings.norm\.bias': {
        'hf_k': 'transformer.word_embeddings_layernorm.bias',
    },
    r'([0-9]+)\.input_layernorm\.weight': {
        'hf_k': 'transformer.h.<LAYER>.input_layernorm.weight'
    },
    r'([0-9]+)\.input_layernorm\.bias': {
        'hf_k': 'transformer.h.<LAYER>.input_layernorm.bias'
    },
    r'([0-9]+)\.self_attention\.query_key_value\.weight': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.query_key_value.weight',
    },
    r'([0-9]+)\.self_attention\.query_key_value\.bias': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.query_key_value.bias',
    },
    r'([0-9]+)\.self_attention\.dense\.weight': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.dense.weight',
    },
    r'([0-9]+)\.self_attention\.dense\.bias': {
         'hf_k': 'transformer.h.<LAYER>.self_attention.dense.bias',
    },
    r'([0-9]+)\.post_attention_layernorm\.weight': {
        'hf_k': 'transformer.h.<LAYER>.post_attention_layernorm.weight',
    },
    r'([0-9]+)\.post_attention_layernorm\.bias': {
        'hf_k': 'transformer.h.<LAYER>.post_attention_layernorm.bias',
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.weight': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_h_to_4h.weight',
    },
    r'([0-9]+)\.mlp\.dense_h_to_4h\.bias': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_h_to_4h.bias',
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.weight': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_4h_to_h.weight',
    },
    r'([0-9]+)\.mlp\.dense_4h_to_h\.bias': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_4h_to_h.bias',
    },
    r'([0-9]+)\.bias': {
        'hf_k': 'transformer.ln_f.bias'
    },
    r'([0-9]+)\.weight': {
        'hf_k': 'transformer.ln_f.weight'
    },
}
