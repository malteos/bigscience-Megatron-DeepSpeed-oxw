

def _add_oxw_args(parser):
    group = parser.add_argument_group('oxw',
                                      'Custom oxw Configurations (load from pretrained hf, ...')
    group.add_argument('--from-pretrained-hf', type=str, default=None,
                       help='Path to pretrained Huggingface model.')
    group.add_argument('--from-pretrained-meg', type=str, default=None,
                       help='Path to pretrained Megatron model (no deepspeed).')
    group.add_argument('--bitfit', action='store_true',
                       help='Enable BitFit (training bias terms only)',
                       dest='bitfit')
    group.add_argument('--no-final-layer-norm', action='store_true',
                       help='Disable final layer norm after transformer layers (as done in OPT)',
                       dest='no_final_layer_norm')
    group.add_argument('--relu-activation', action='store_true',
                       help='Enable ReLu activation function (as done in BART/OPT)',
                       dest='relu_activation')
    # group.add_argument('--position-embedding-offset', type=int, default=None,
    #                    help='Offset to position embeddings (as done in OPT).')
    return parser
