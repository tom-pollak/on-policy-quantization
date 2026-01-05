# Workshop Paper Plan

## Paper Outline

### 1. Introduction
- KD for compression: teacher → smaller student
- Problem: distribution mismatch (student trains on fixed data, doesn't learn from own mistakes)
- GKD (Agarwal et al., 2023) proposes on-policy distillation: student generates, learns from teacher feedback
- **Our question**: Does on-policy help for *quantization* (same weights, different precision)?
- **Spoiler**: No. Null result. Distribution mismatch may not matter when weights are shared.

### 2. Background
- Standard KD: minimize KL(teacher || student) on fixed dataset
- GKD framework (cite Agarwal et al.):
  - λ: interpolation between off-policy (λ=0) and on-policy (λ=1)
  - β: interpolation between forward KL (β=0) and reverse KL (β=1)
- Quantization context: INT4 student shares FP16 teacher weights

### 3. Experimental Setup
- Model: Qwen3-4B (can extend to other architectures)
- Quantization: INT4 via torchao (can compare to GPTQ/AWQ)
- Training data: allenai/tulu-3-sft-mixture
- Implementation: TRL GKDTrainer
- Benchmarks: HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, MMLU
- Hyperparameters swept: λ, β, LR, batch size, rollout length, training steps

### 4. Results
#### 4.1 Main Result: On-Policy vs Off-Policy
- Table: Best λ=0 vs λ=0.5 vs λ=1 (from README)
- Finding: ~0.3% difference, within noise

#### 4.2 Learning Rate
- Figure: LR sweep for λ=0 and λ=1
- Finding: Lower LRs generalize better despite higher training loss
- Discussion: "overfitting" on training distribution, not data

#### 4.3 Extended Training
- Table: 1k vs 20k steps
- Finding: 42% loss reduction, flat eval accuracy → hit INT4 ceiling

#### 4.4 Batch Size & Rollout Length (λ=1)
- Tables from README
- Finding: Not sensitive parameters

### 5. Discussion
- Why null result? Hypothesis: distribution mismatch is minimal when student/teacher share weights
- Standard KD: different architectures → student distribution far from teacher
- Quantization: same weights → student already close to teacher distribution
- Practical implication: skip expensive on-policy rollouts for quantization

### 6. Conclusion
- On-policy distillation doesn't help for quantization
- Simpler off-policy SFT is sufficient and cheaper
- Future work: test on other quantization levels (INT8, INT2), generation tasks

---

## Appendix (β finding)
- Table: β sweep for λ=0 and λ=1
- Observation: Forward KL (β=0) slightly better for off-policy, reverse KL (β=1) theoretically motivated for on-policy but empirically similar
- Margins too small to draw strong conclusions

---

## Key Citations
- GKD: Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (2023)
- torchao quantization
- lm-evaluation-harness (benchmarks)

---

## TODOs before submission
- [ ] Add compute cost comparison (on-policy vs off-policy)
- [ ] Run multiple seeds for error bars (if time)
- [ ] Consider adding a generation benchmark
- [ ] (Optional) Test additional model architectures
- [ ] (Optional) Compare torchao INT4 vs GPTQ vs AWQ
