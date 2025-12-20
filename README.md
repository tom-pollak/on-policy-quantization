# On-Policy Distillation

## Setup

Install dependencies and lock the environment with `uv`:

```bash
uv sync
```

torch transformers datasets accelerate bitsandbytes peft scipy pydantic pydantic_config

## Training

```bash
# avg completion length ~400
k8s/train.sh lmbda_1_int4 --lmbda 1 --quant-type int4 --max-new-tokens 128

k8s/train.sh lmbda_0_int4 --lmbda 0 --quant-type int4

k8s/train.sh lmbda_05_int4 --lmbda 0.5 --quant-type int4 --max-new-tokens 128

k8s/train.sh lmbda_1_bnb_fp4 --lmbda 1 --quant-type bnb_fp4 --max-new-tokens 128

k8s/train.sh lmbda_0_bnb_fp4 --lmbda 0 --quant-type bnb_fp4
```

```bash
k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_1_int4 --lora_paths dump/lmbda_0_int4 --lora_paths dump/lmbda_05_int4 --lora_paths dump/lmbda_1_bnb_fp4 --lora_paths dump/lmbda_0_bnb_fp4



k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_0_int4 --lora_paths dump/lmbda_0_bnb_fp4

# tomorrow
k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_1_int4 --lora_paths dump/lmbda_05_int4 --lora_paths dump/lmbda_1_bnb_fp4 --no-eval-teacher
```

### 1. Off-policy KD baseline

```bash
uv run accelerate launch train.py --lmbda 0 --output_dir qwen_kd_baseline --quant-type int4
```

### 2. On-policy KD

```bash
uv run accelerate launch train.py --lmbda 1 --output_dir qwen_onpolicy_4b_int4 --quant-type int4
```

### 3. Perplexity comparison (teacher vs PTQ vs KD vs on-policy KD)

```bash
uv run python eval.py
```

## Results

```
model                     | hellaswa | arc_easy | arc_chal | winogran |     mmlu
teacher                   |   0.5262 |   0.8308 |   0.5572 |   0.6835 |   0.7067
ptq_bnb_nf4               |          |          |   0.5410 |          |
ptq_int4                                        |   0.5265
int4_lmbda_0                                    |   0.5375
int4_lmbda_1                                    |   0.5384
int4_lmbda_05                                   |   0.5427
bnb_fp4_lmbda_1                                 |   0.5392
```
