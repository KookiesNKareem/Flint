import torch
import torch.nn.functional as F
import flint_gpu as flint

@flint.fast
def recurrence(q, k, v, decay, beta, transfer, state, scale):
    """Delta writes to fast memory; a state-dependent gate consolidates slow memory."""
    q, k, v, decay, beta, transfer = (
        x.to(torch.promote_types(x.dtype, torch.float32))
        for x in (q, k, v, decay, beta, transfer)
    )
    k = F.normalize(k, dim=-1)
    decay = F.logsigmoid(decay)
    beta = beta.sigmoid()
    fast, slow = state['fast'], state['slow']
    outputs = []
    for t in range(q.shape[2]):
        key, value = k[:, :, t], v[:, :, t]
        fast = fast * decay[:, :, t].exp()[..., None]
        prediction = (fast * key[..., None]).sum(-2)
        fast = fast + key[..., None] * (
            beta[:, :, t, None] * (value - prediction)
        )[..., None, :]
        agreement = (fast * slow).mean(dim=(-2, -1))
        gate = torch.sigmoid(transfer[:, :, t] + 0.05 * agreement)[..., None, None]
        slow = (1 - gate) * slow + gate * fast
        outputs.append(((fast + slow) * (0.5 * scale * q[:, :, t])[..., None]).sum(-2))
    return torch.stack(outputs, dim=2), {'fast': fast, 'slow': slow}

