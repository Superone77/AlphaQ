import torch
import torch.nn.functional as F


@torch.no_grad()
def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')
