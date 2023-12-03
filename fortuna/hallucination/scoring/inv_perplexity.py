import torch
from torch.nn import CrossEntropyLoss


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor, n_final_tokens: int):
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return torch.exp(
        loss_fct(shift_logits.transpose(1, 2), shift_labels)[:, -n_final_tokens:].mean(
            1
        )
    )


@torch.no_grad()
def inv_perplexity(logits: torch.Tensor, labels: torch.Tensor, n_final_tokens: int):
    return 1 / perplexity(logits=logits, labels=labels, n_final_tokens=n_final_tokens)
