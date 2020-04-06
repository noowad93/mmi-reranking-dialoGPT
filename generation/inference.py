from torch.autograd import backward
from generation.config import InferenceConfig
from typing import List, Dict
import torch
import torch.nn.functional as F
from generation.data import GPTDataset
from torch import masked_fill
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class Inferencer:
    def __init__(
        self,
        config: InferenceConfig,
        tokenizer: GPT2Tokenizer,
        forward_model: GPT2LMHeadModel,
        backward_model: GPT2LMHeadModel,
    ):
        self.config: InferenceConfig = config
        self.forward_model: GPT2LMHeadModel = forward_model
        self.forward_model.to(config.device_for_forward)
        self.forward_model.eval()
        self.forward_device: torch.device = torch.device(self.config.device_for_forward)

        self.backward_model: GPT2LMHeadModel = backward_model
        self.backward_model.to(config.device_for_backward)
        self.backward_model.eval()
        self.backward_device: torch.device = torch.device(self.config.device_for_backward)
        self.tokenizer: GPT2Tokenizer = tokenizer

    def run(self, contexts: List[str]):
        dataset = GPTDataset(contexts, self.tokenizer, self.config.max_context_length, self.config.eos_token_idx)
        results = []
        for forward_context, backward_context in dataset:
            forward_context = forward_context.to(self.forward_device)
            backward_context = backward_context.to(self.backward_device)

            reply_dict = self._get_replies(forward_context, backward_context)
            # mmi reranking
            sorted_replies = [i[0] for i in sorted(reply_dict.items(), key=lambda x: x[1])][:5]
            results.append(sorted_replies)
        return results

    def _get_replies(self, forward_context: torch.Tensor, backward_context: torch.Tensor) -> Dict[str, float]:
        # inference w.r.t context
        _, past = self.forward_model.forward(forward_context[:, :-1], past=None)

        # auto-regressive inference
        reply_dict = {}
        for _ in range(self.config.num_samples):
            model_output = torch.tensor([[]], dtype=torch.long).to(self.forward_device)
            forward_output_token = forward_context[:, -1:]
            while True:
                forward_output_token, past = self.forward_model.forward(forward_output_token, past)
                forward_output_token = forward_output_token[:, -1, :].float()
                ignore_indices = forward_output_token < torch.topk(forward_output_token, self.config.top_k)[0][
                    :, -1
                ].unsqueeze(1)
                forward_output_token[ignore_indices] = -float("inf")
                forward_output_token = torch.multinomial(F.softmax(forward_output_token/self.config.temperature, dim=-1), num_samples=1)
                model_output = torch.cat((model_output, forward_output_token), dim=1)
                if forward_output_token.item() == self.config.eos_token_idx:
                    break
            backward_model_score = self._get_backward_score(
                model_output.to(self.config.device_for_forward), backward_context
            )
            reply = self.tokenizer.decode(model_output.tolist()[0], skip_special_tokens=True)
            reply_dict[reply] = backward_model_score
        return reply_dict

    def _get_backward_score(self, forward_model_output: torch.Tensor, backward_context: torch.Tensor) -> float:
        forward_model_output = forward_model_output.to(self.config.device_for_backward)

        backward_model_input = torch.cat((forward_model_output, backward_context), dim=1)
        # ignore_index = -100
        mask = torch.full_like(forward_model_output, -100, dtype=torch.long)
        target_labels = torch.cat((mask, backward_context), dim=1)
        loss, _, _ = self.backward_model.forward(backward_model_input, labels=target_labels)
        return loss.float().item()
