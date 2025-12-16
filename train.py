import math
import os
import random

os.environ["WANDB_PROJECT"] = "on-policy-distillation"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl.experimental.gkd import GKDConfig, GKDTrainer

from config import OnPolicyKDConfig


@validate_call
def main(conf: OnPolicyKDConfig = OnPolicyKDConfig()) -> None:
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)

    dtype = torch.bfloat16 if conf.mixed_precision == "bf16" else torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset - split into train/eval (fixed seed for reproducible eval set)
    raw_dataset = load_dataset(conf.dataset_name, split="train")
    split_dataset = raw_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )

    # Student model (quantized + LoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    student_model = AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        quantization_config=bnb_config,
        device_map={"": local_rank},
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # GKD training config
    training_args = GKDConfig(
        output_dir=str(conf.output_dir),
        per_device_train_batch_size=conf.per_device_train_batch_size,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        learning_rate=conf.learning_rate,
        max_steps=conf.max_train_steps,
        warmup_ratio=conf.warmup_ratio,
        logging_steps=conf.logging_steps,
        save_steps=conf.save_steps,
        eval_strategy="steps",
        eval_steps=conf.eval_steps,
        bf16=conf.mixed_precision == "bf16",
        fp16=conf.mixed_precision == "fp16",
        report_to=["wandb"],
        run_name="on_policy_gkd",
        ddp_find_unused_parameters=False,
        # TODO: revisit gradient checkpointing - DDP + LoRA + checkpointing causes
        # "parameter marked ready twice" error even with use_reentrant=False.
        # Options: FSDP, single-GPU, or fix upstream in TRL/PEFT.
        gradient_checkpointing=False,
        lmbda=conf.lmbda,  # 0.0 = off-policy (dataset), 1.0 = on-policy (student)
        beta=conf.beta,  # 0.0 = forward KL, 1.0 = reverse KL
        temperature=conf.temperature,
        max_new_tokens=conf.max_new_tokens,
    )

    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(conf.output_dir))
    tokenizer.save_pretrained(str(conf.output_dir))


if __name__ == "__main__":
    main(**parse_argv())
