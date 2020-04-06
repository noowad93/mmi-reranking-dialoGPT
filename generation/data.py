from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from typing import List, Tuple
import torch


class GPTDataset(Dataset):
    def __init__(self, contexts: List[str], tokenizer: GPT2Tokenizer, max_seq_len: int, eos_token_idx: int):

        self._contexts: List[str] = contexts
        self.max_seq_len: int = max_seq_len
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.eos_token_idx = eos_token_idx

    def __len__(self) -> int:
        return len(self._contexts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._create_data_instance(self._contexts[index])

    def _create_data_instance(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_list = []

        context = input_text.split("<|endoftext|>")
        for message in context:
            if message == "":
                continue
            input_token = self.tokenizer.encode(message, return_tensors="pt")
            input_token = torch.cat((input_token, torch.tensor([[self.eos_token_idx]])), dim=1)
            input_list.append(input_token)

        # Truncate when total message length is bigger than max sequence length
        total_length = 0
        for i, message in enumerate(reversed(input_list)):
            total_length += message.shape[1]
            if total_length > self.max_seq_len:
                input_list = input_list[-i:]
                break

        forward_input_list = torch.cat(input_list, dim=1)
        backward_input_list = torch.cat(list(reversed(input_list)), dim=1)
        return forward_input_list, backward_input_list
