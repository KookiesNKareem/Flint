"""Run one small, deterministic PyTorch training experiment on the CPU."""
import torch
import flint


@flint.fast
def recurrence(x, state):
    outputs = []
    for sample in x.unbind(1):
        state = torch.tanh(state * 0.8 + sample)
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def main():
    torch.manual_seed(7)
    x = torch.randn(4, 16, 8)
    scale = torch.nn.Parameter(torch.ones(8))
    optimizer = torch.optim.AdamW([scale], lr=0.05)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        y = recurrence(x * scale, torch.zeros_like(x[:, 0]))
        loss = y.square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses))) and losses[-1] < losses[0]
    print(f"6 CPU updates, loss {losses[0]:.5f} -> {losses[-1]:.5f}")


if __name__ == "__main__":
    main()
