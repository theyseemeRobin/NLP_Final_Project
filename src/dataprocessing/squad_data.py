import pandas as pd
import torch
from torch.utils.data import Dataset


class SquadData(Dataset):
    def __init__(self, json_path, tokenizer, max_tokens=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Parameters
        ----------
        json_path : str
            Path to json file
        tokenizer : transformers.PreTrainedTokenizerFast
        max_tokens : int
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device

    def __getitem__(self, idx):
        """
        For a given datapoint `idx`, return the sequence of token ids (input_ids), the attention mask, the start
        position of the answer and the end position of the answer as tensors.
        """
        return {
            'input_ids': torch.randint(0, 100, (self.max_tokens,), device=self.device),
            'attention_mask': torch.randint(0, 1, (self.max_tokens,), device=self.device),
            'start_positions': torch.randint(0, self.max_tokens, (1,), device=self.device).squeeze(0),
            'end_positions': torch.randint(0, self.max_tokens, (1,), device=self.device).squeeze(0)
        }

    def __len__(self):
        """
        Return number of samples in the processed dataset
        """
        return 500