from torch import nn

from megatron import print_rank_0


def deactivate_gradients(model: nn.Module, needle_name: str = 'bias'):
    """
    Turn off the model parameters requires_grad except the ones matching the needle term (e.g., bias terms)
    (aka "BitFit" https://arxiv.org/pdf/2106.10199v2.pdf)
    """
    deactivated = []
    activated = []

    for name, param in model.named_parameters():
        if needle_name in name:
            param.requires_grad = True
            activated.append(name)
        else:
            param.requires_grad = False
            deactivated.append(name)

    print_rank_0(f'Activated parameters (name contains `{needle_name}`): {len(activated)} ({activated[:10]} ...)')
    print_rank_0(f'Deactivated parameters: {len(deactivated)} ({deactivated[:3]} ...)')

    return model
