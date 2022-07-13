import json

from megatron import get_args, print_rank_0
from megatron.checkpointing import save_checkpoint
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.utils import get_parameters_in_billions

from pretrain_gpt import model_provider as gpt_model_provider

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn


@record
def main():
    """

    A simple script to initialize a model and directly save its checkpoint to disk (without any training or evaluation).

    This is a modified version of `megatron.training.pretrain` (stripped from all unnecessary code)

    :return:
    """

    args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
    extra_args_provider = None

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()

    if args.save is None:
        raise ValueError('--save is not set')

    model_provider = gpt_model_provider

    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    args.parameters_in_billions_no_embedding = get_parameters_in_billions(model, exclude_embeddings=True)
    print_rank_0(f'estimated model parameters: {get_parameters_in_billions(model)}')
    print_rank_0(f'estimated model parameters without embeddings: {get_parameters_in_billions(model, exclude_embeddings=True)}')

    # Print setup timing.
    print_rank_0('done with setup ...')

    save_checkpoint(0, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()

