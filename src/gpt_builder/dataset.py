from typing import Iterable
import torch
from torch.utils.data import Dataset

class BigramDataset(Dataset):
    """
    Constructor for building a bigram torch Dataset.
    """
    def __init__(self, doc_tokens: Iterable[int], block_size: int = 8) -> None:
        super().__init__()
        self.doc_tokens = doc_tokens
        if isinstance(doc_tokens, torch.Tensor):
            self.doc_tokens = self.doc_tokens.clone().detach()
        self.block_size = block_size
    
    def __len__(self) -> int:
        """
        Number of data points.
        
        Args
        ----
        None
        
        Return
        ------
            int: Length of dataset
        """
        return len(self.doc_tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get bigram input/output tensors for LLM training and/or evaluation.
        
        Args
        ----
            idx: int
                Index of dataset
        Returns
        -------
            tuple[torch.Tensor, torch.Tensor]
                Input and ouput bigram tensors.
        """
        return self.doc_tokens[idx:idx+self.block_size], self.doc_tokens[idx+1:idx+self.block_size+1]