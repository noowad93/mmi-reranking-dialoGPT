import sys
from logging import StreamHandler
import logging

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from generation.utils import load_model_weight
from generation.config import InferenceConfig
from generation.inference import Inferencer


def main():
    # Config
    config = InferenceConfig()
    gpt_config = GPT2Config.from_json_file(config.model_config_path)

    # torch related
    torch.set_grad_enabled(False)
    torch.manual_seed(config.random_seed)

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)

    # Text Utils
    logging.info(f"loading Tokenizer...")
    tokenizer = GPT2Tokenizer(config.tokenizer_vocab_path, config.tokenizer_merge_path)

    # Forward Model
    logging.info(f"loading Forward Model...")
    forward_model = GPT2LMHeadModel(gpt_config)
    forward_model.load_state_dict(load_model_weight(gpt_config, config.forward_model_path))

    # Backward Model
    logging.info(f"loading Backward Model...")
    backward_model = GPT2LMHeadModel(gpt_config)
    backward_model.load_state_dict(load_model_weight(gpt_config, config.backward_model_path))

    # Example
    example_contexts = [
        "<|endoftext|>".join(["How are you doing?"]),
        "<|endoftext|>".join(["Does money buy happiness?"]),
        "<|endoftext|>".join(["Does money buy happiness?", "Depends how much money you spend on it .",]),
        "<|endoftext|>".join(
            [
                "Does money buy happiness?",
                "Depends how much money you spend on it .",
                "What is the best way to buy happiness ?",
            ]
        ),
    ]
    inferencer = Inferencer(config, tokenizer, forward_model, backward_model)
    results = inferencer.run(example_contexts)

    for context, results in zip(example_contexts, results):
        logging.info(f"Example Context:{context}")
        for i, reply in enumerate(results):
            logging.info(f"Output Utterance Top-{i+1}: {reply}")


if __name__ == "__main__":
    main()
