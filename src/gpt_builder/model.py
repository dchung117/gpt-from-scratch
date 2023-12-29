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

class GPTLanguageModel(nn.Module):
    """
    Decoder-only large language model.

    Args
    ----
    vocab_size: int
        Number of tokens in vocabulary
    block_size: int
        Number of tokens per example
    n_decoders: int (def. 4)
        Number of decoder modules
    n_heads: int (def. 8)
        Number of attention heads per decoder
    d_embed: int (def. 384)
        Dimensionality of embeddings
    """
    def __init__(self, vocab_size: int, block_size: int,
        n_decoders: int = 4, n_heads: int = 8, d_embed: int = 384) -> None:
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.pos_embedding = nn.Embedding(block_size, d_embed)
        self.decoders = nn.Sequential(
            *[DecoderBlock(d_embed, n_heads) for _ in range(n_decoders)]
        )
        self.layer_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, vocab_size)

class DecoderBlock(nn.Module):
    """
    Decoder block for GPT language model.
    """
    pass