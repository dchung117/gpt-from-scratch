import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLLM(nn.Module):
    """
    Bigram large language model.
    """
    def __init__(self) -> None:
        super().__init__()