import torch
from torch.nn import CrossEntropyLoss


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor, init_pos: int = 0):
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    perplexities = torch.exp(
        loss_fct(shift_logits.transpose(1, 2), shift_labels)[:, init_pos:].mean()
    )
    return torch.mean(perplexities)


@torch.no_grad()
def inv_perplexity(logits: torch.Tensor, labels: torch.Tensor, init_pos: int = 0):
    return 1 / perplexity(logits=logits, labels=labels, init_pos=init_pos)
