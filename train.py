import os

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

from config import TrainConfig, Tee


@validate_call
def main(cfg: TrainConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    Tee.redirect_stdout_stderr(cfg.output_dir / "train.log")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_JOB_TYPE"] = "train"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg.dataset_name, split="train")
    dataset = dataset.filter(
        lambda x: len(x.get("messages", [])) > 0
        and all(m.get("content", "").strip() for m in x["messages"] if m.get("role") == "user")
    )

    # Models
    teacher = cfg.load_model()
    student = cfg.load_quant_model("qat")
    if cfg.use_lora:
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
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    training_args = GKDConfig(
        bf16=cfg.mixed_precision == "bf16",
        fp16=cfg.mixed_precision == "fp16",
        # torch_compile=True,
        # torch_compile_backend="inductor",
        report_to=["wandb"],
        run_name=cfg.output_dir.stem,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        **cfg.trainer_kwargs(),
    )

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))


if __name__ == "__main__":
    main(TrainConfig(**parse_argv()))
