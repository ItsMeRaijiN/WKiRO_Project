import torch


def reconstruction_loss(x, x_hat, derivative_weight: float = 0.5):
    rec_loss = torch.mean((x - x_hat) ** 2)
    if x.shape[1] < 2:
        return rec_loss

    x_diff = x[:, 1:, :] - x[:, :-1, :]
    x_hat_diff = x_hat[:, 1:, :] - x_hat[:, :-1, :]
    derivative_loss = torch.mean((x_diff - x_hat_diff) ** 2)
    return rec_loss + derivative_weight * derivative_loss


def _unwrap_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def train_epoch(model, loader, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0

    if len(loader) == 0:
        raise ValueError("Training loader is empty.")

    for batch in loader:
        x = _unwrap_batch(batch).to(device)

        optimizer.zero_grad(set_to_none=True)
        x_hat = model(x)

        loss = reconstruction_loss(x, x_hat)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    if len(loader) == 0:
        raise ValueError("Validation loader is empty.")

    for batch in loader:
        x = _unwrap_batch(batch).to(device)
        x_hat = model(x)
        total_loss += reconstruction_loss(x, x_hat).item()

    return total_loss / len(loader)