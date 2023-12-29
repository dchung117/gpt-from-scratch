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
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.pos_embedding = nn.Embedding(block_size, d_embed)
        self.decoders = nn.Sequential(
            *[DecoderBlock(d_embed, n_heads) for _ in range(n_decoders)]
        )
        self.layer_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, vocab_size)

        self.apply(self.__init_wts)

    def __init_wts(self, module: nn.Module) -> None:
        """
        Initializes submodule weights.

        Args
        ----
            module: nn.Module
                Module whose weights are initialized
        Return
        ------
            None
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idxs: torch.Tensor, tgts: torch.Tensor | None) -> torch.Tensor:
        x_tokens = self.token_embedding(idxs)  # (B, T, d_embed)
        x_pos = self.pos_embedding(torch.arange(self.block_size, device=x_tokens.device))  # (T, d_embed)
    
        x = x_tokens + x_pos  # (B, T, d_embed)
        x = self.decoders(x)  # (B, T, d_embed)
        x = self.layer_norm(x)  # (B, T, d_embed)
        return self.head(x)  # (B, T, vocab_size)

class DecoderBlock(nn.Module):
    """
    Decoder block for GPT language model.

    Args
    ----
        d_embed: int
            Dimensionality of embeddings
        n_heads: int
            Number of attention heads
    """
    def __init__(self, d_embed: int, n_heads: int) -> None:
        super().__init__()
        d_head = d_embed // n_heads   # dimensionality for each head
        self.attn = MultiHeadAttention(n_heads, d_head, d_embed)
        self.ff = FeedForward(d_embed)
        self.layer_norm_1 = nn.LayerNorm(d_embed)
        self.layer_norm_2 = nn.LayerNorm(d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  Attention, residual
        y = self.attn(x)
        x = self.layer_norm_1(x + y)

        #  Feed forward, residual
        y = self.ff(x)
        return self.layer_norm_2(x + y)

class MultiHeadAttention(nn.Module):
    """
    Multi-headed attention module
    
    Args
    ----
        n_heads: int
            Number of attention heads
        d_head: int
            Dimensionality of each attention head
        d_embed: int
            Dimensionality of embedding
        p_dropout: float (def. 0.2)
            Dropout hyperparameter
    """
    def __init__(self, n_heads: int, d_head: int, d_embed: int, p_dropout: float = 0.2) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(d_head) for _ in range(n_heads)]
        )
        self.ff = nn.Linear(n_heads * d_head, d_embed)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [h(x) for h in self.heads], dim=-1  # (B, T, n_heads * d_head)
        )
        x = self.ff(x)
        return self.dropout(x)

class AttentionHead(nn.Module):
    """
    Attention head module for GPT LLM
    """
    pass

class FeedForward(nn.Module):
    """
    Feedforward GPT LLM module

    
    Args
    ----
        d_embed: int
            Dimensionality of embeddings
        p_dropout: float (def. 0.2)
            Dropout hyperparameter
    """
    def __init__(self, d_embed: int, p_dropout: float = 0.2) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_embed, 4*d_embed),
            nn.ReLU(),
            nn.Linear(4*d_embed, d_embed),
            nn.Dropout(p=p_dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)