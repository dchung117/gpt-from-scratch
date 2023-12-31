import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

def train_step(model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer
    ) -> None:
    """
    Executes a single training step

    Args
    ----
        model: torch.nn.Module
            Large language model to be trained.
        inputs: torch.Tensor
            Input tokens for training (shape: (B, T, vocab_size)).
        targets: torch.Tensor
            Target tokens for training (shape: (B, T, vocab_size)).
        optimizer: torch.optim.Optimizer
            Optimization algorithm for training.
    Returns
    -------
        float:
            Training loss over the training batch
    """
    model.train()
    logits = model(inputs)

    optimizer.zero_grad()
    loss = bigram_crossentropy_loss(logits, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def get_loss(dl: DataLoader,
    model: nn.Module,
    device: torch.device
    ) -> float:
    """
    Estimate loss.

    Args
    ----
        dl: DataLoader
            Dataloader containing bigrams
        model: nn.Module
            Large language model
        device: torch.device
            Model device
    
    Return
    ------
        float: 
            Loss
    """
    losses = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in dl:
            logits = model(inputs.to(device)).cpu()
            loss = bigram_crossentropy_loss(logits, targets).item()
            losses.append(loss)

    return sum(losses)/len(losses)