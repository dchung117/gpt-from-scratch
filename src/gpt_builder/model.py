import torch
import torch.nn as nn

class BigramLLM(nn.Module):
    """
    Bigram large language model.

    Args
    ----
    vocab_size: int
        Number of tokens in vocabulary
    """
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embed = self.embedding(x)
        return x_embed
