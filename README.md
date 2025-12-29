# On-Policy Quantization

Knowledge distillation from FP32 teacher to MXFP4 student using on-policy (student-generated) data rather than static datasets.

Standard KD trains on fixed datasets, causing distribution mismatch—the student never learns from its own mistakes. On-policy distillation fixes this by having the student generate sequences and learning from teacher feedback on those generations. Inspired by [GKD](https://arxiv.org/abs/2306.13649) and [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).

- `λ` controlling the interpolation: `λ=0` is off-policy (dataset only), `λ=1` is fully on-policy (student generations only).

## Setup

```bash
uv sync
```

## Training

```bash
# Off-policy KD baseline (λ=0)
uv run accelerate launch train.py --lmbda 0 --output_dir qwen_kd_baseline --quant-type int4 --do-eval

# On-policy KD (λ=1)
uv run accelerate launch train.py --lmbda 1 --output_dir qwen_onpolicy_4b_int4 --quant-type int4 --do-eval
```

```
model                     | hellaswa | arc_easy | arc_chal | winogran |     mmlu
teacher                   |   0.5262 |   0.8308 |   0.5572 |   0.6835 |   0.7067

lmbda0_lr5e5_steps20k     |   0.5126 |   0.8199 |   0.5461 |   0.6638 |   0.6735

lmbda1_bs32               |   0.5119 |   0.8224 |   0.5401 |   0.6709 |   0.6839
lmbda1_bs16               |   0.5131 |   0.8173 |   0.5427 |   0.6732 |   0.6847

lmbda0_lr5e6              |   0.5149 |   0.8165 |   0.5375 |   0.6875 |   0.6822
lmbda0_lr1e5              |   0.5153 |   0.8178 |   0.5358 |   0.6780 |   0.6807
lmbda0_lr2e5              |   0.5177 |   0.8157 |   0.5486 |   0.6717 |   0.6797
lmbda0_lr5e5              |   0.5137 |   0.8148 |   0.5341 |   0.6725 |   0.6773
lmbda0_lr1e4              |   0.5127 |   0.8106 |   0.5213 |   0.6701 |   0.6742


lmbda0_beta1              |   0.5145 |   0.8203 |   0.5384 |   0.6835 |   0.6788
lmbda0_beta05             |   0.5172 |   0.8220 |   0.5367 |   0.6788 |   0.6811
lmbda0_beta0              |   0.5213 |   0.8232 |   0.5401 |   0.6780 |   0.6806

lmbda1_beta05             |   0.5150 |   0.8194 |   0.5384 |   0.6748 |   0.6864

lmbda1_lr5e5              |   0.5118 |   0.8123 |   0.5256 |   0.6646 |   0.6776
lmbda1_lr1e4              |   0.5105 |   0.8157 |   0.5367 |   0.6701 |   0.6769

lmbda1_tok256             |   0.5127 |   0.8178 |   0.5299 |   0.6811 |   0.6824
lmbda1_tok512             |   0.5141 |   0.8148 |   0.5358 |   0.6756 |   0.6845

lmbda_1_int4              |   0.5121 |   0.8186 |   0.5333 |   0.6764 |   0.6855
lmbda_05_int4             |   0.5160 |   0.8194 |   0.5333 |   0.6803 |   0.6857
lmbda_1_int4              |   0.5121 |   0.8186 |   0.5333 |   0.6764 |   0.6855
```
