import torch
import torch.nn.functional as F

def bigram_crossentropy_loss(preds: torch.Tensor, tgts: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss for bigram LLM.

    Args
    ----
        preds: torch.Tensor
            Bigram LLM logits (shape: (*, vocab_size))
        tgts: torch.Tensor:
            Targets vocab IDs (shape: (*,))
    Return
    ------
        torch.Tensor
            Cross entropy loss
    """
    vocab_size = preds.shape[-1]
    preds = preds.reshape(-1, vocab_size)
    tgts = tgts.reshape(-1)
    return F.cross_entropy(preds, tgts)