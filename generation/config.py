from typing import NamedTuple


class InferenceConfig(NamedTuple):
    random_seed: int = 0

    max_context_length: int = 128
    num_samples: int = 10
    top_k: int = 10
    mmi_temperature: float = 0.5

    tokenizer_vocab_path: str = "pretrained_gpt/vocab.json"
    tokenizer_merge_path: str = "pretrained_gpt/merges.txt"
    eos_token_idx: int = 50256

    device_for_forward: str = "cuda:1"
    device_for_backward: str = "cuda:2"

    model_config_path: str = "pretrained_gpt/config.json"

    forward_model_path: str = "pretrained_gpt/medium_forward.pkl"
    backward_model_path: str = "pretrained_gpt/medium_backward.pkl"
