"""Evaluate GPTQ and AWQ quantized models for comparison with torchao distillation."""

import os

os.environ.setdefault("HF_HOME", "./hf-cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb
from datasets import Dataset, load_dataset
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier

from config import QuantEvalConfig, Tee
from eval import create_eval_table, eval_and_log


def get_calibration_dataset(cfg: QuantEvalConfig, tokenizer):
    """Prepare calibration data for quantization."""
    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)
    examples = []
    for i, sample in enumerate(ds):
        if i >= cfg.num_calibration_samples:
            break
        if "messages" in sample:
            text = tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
        elif "text" in sample:
            text = sample["text"]
        else:
            text = str(sample[list(sample.keys())[0]])

        if len(text) > 256:
            examples.append({"text": text[: cfg.max_seq_length * 4]})

    ds = Dataset.from_list(examples)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=cfg.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def quantize_gptq(cfg: QuantEvalConfig, tokenizer, calib_ds):
    """Quantize model with GPTQ via llmcompressor and save."""
    print("Quantizing with GPTQ INT4")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.dtype, device_map="auto"
    )
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=cfg.max_seq_length,
        num_calibration_samples=cfg.num_calibration_samples,
    )
    return model


def quantize_awq(cfg: QuantEvalConfig, tokenizer, calib_ds):
    """Quantize model with AWQ via llmcompressor and save."""
    print("Quantizing with AWQ INT4")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.dtype, device_map="auto"
    )
    recipe = AWQModifier(
        targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"], duo_scaling="both"
    )
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=cfg.max_seq_length,
        num_calibration_samples=cfg.num_calibration_samples,
    )
    return model


def main(cfg: QuantEvalConfig):
    wandb.init(project=cfg.wandb_project, name="gptq-awq-eval", tags=cfg.tags)
    Tee.redirect_stdout_stderr("./gptq_awq.log")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    columns, table = create_eval_table(cfg.tasks)
    header = " | ".join(f"{c[:8]:>8s}" for c in columns)
    print(header)

    calib_ds = get_calibration_dataset(cfg, tokenizer)
    methods = [("GPTQ", quantize_gptq), ("AWQ", quantize_awq)]

    for name, quantize_fn in methods:
        model = quantize_fn(cfg, tokenizer, calib_ds)
        eval_and_log(name, model, tokenizer, cfg.tasks, columns, table)
        del model
        torch.cuda.empty_cache()

    wandb.log({"eval_results": table})
    wandb.summary["eval_results"] = table
    wandb.finish()


if __name__ == "__main__":
    main(parse_argv())
