import torch
from typing import Dict


def load_model_weight(config, model_file_path) -> Dict[str, float]:
    weights = torch.load(model_file_path)
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)
    return weights
