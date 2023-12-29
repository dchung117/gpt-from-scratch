import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLLM(nn.Module):
    """
    Bigram large language model. Maps each input token to logits for each element in vocabulary.

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

    def generate(self, x: torch.Tensor, n_gen: int) -> torch.Tensor:
        """
        Generate new tokens for given input.

        Args
        ----
            x: torch.Tensor
                Context token sequence (shape: (*, vocab_size))
            n_gen: int
                Number of new tokens to generate
        Return
        ------
            torch.Tensor:
                Sequence containing input context and generated tokens
        """
        if len(x.shape) == 1: # Unsqueeze to b, t
            x = x.unsqueeze(0)

        for _ in range(n_gen):
            logits = self(x)[:, -1, :] # Only need to get predictions for last token
            probs = F.softmax(logits, dim=-1)

            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)

        return x

