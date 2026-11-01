import torch
import flint_gpu as flint


@flint.fast
def recurrence(x, h, c):
    """Two coupled states, with a nonlinear update and a slower memory."""
    outputs=[]
    for t in range(x.shape[1]):
        h=torch.tanh(x[:,t]+0.8*h+0.2*c)
        c=0.9*c+0.1*h
        outputs.append(h)
    return torch.stack(outputs,dim=1),h,c
