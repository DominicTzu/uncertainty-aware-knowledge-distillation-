# Uncertainty-Aware Knowledge Distillation for Small Instruction-Tuned Language Models

This project builds a compact and reproducible teacher–student distillation pipeline for instruction-tuned language models, and studies whether **teacher uncertainty** can be used to make token-level knowledge distillation more robust.

The project includes three training settings:

- **Response-only SFT**
- **Vanilla token-level KD**
- **Uncertainty-aware KD (UKD)**

The central idea is simple:

> standard KD treats all target tokens equally, while UKD reduces the distillation weight on tokens where the teacher itself is uncertain.

---

## 1. Project Overview

We use a larger instruction-tuned model as the **teacher** and a smaller model as the **student**, and compare three training strategies:

1. **SFT baseline**: supervised fine-tuning with response-only cross-entropy
2. **Vanilla KD**: SFT + token-level distillation from teacher soft targets
3. **UKD**: SFT + token-level distillation with entropy-based uncertainty weighting

The goal is not to introduce a highly complex method, but to build a clean and interpretable pipeline that can answer the following question:

> Can uncertainty-aware weighting preserve the benefits of knowledge distillation while reducing degradation on difficult, high-uncertainty examples?

---

## 2. Models

### Teacher
- **Qwen2.5-7B-Instruct**

### Student
- **Qwen2.5-3B-Instruct**

### Why this pair
This teacher–student pair was chosen because:

- both models belong to the same family,
- their instruction-following behavior is reasonably aligned,
- implementation is simpler than cross-family distillation,
- the size gap is large enough to make distillation meaningful,
- the setup is practical for high-memory or multi-GPU training environments.

---

## 3. Data Construction

### 3.1 Data source
We use the **UltraChat 200k** dataset as the raw instruction-following corpus.

### 3.2 Sampling
From the original dataset, we first sampled:

- **100,000 candidate training dialogues** from `train_sft`
- **10,000 candidate evaluation dialogues** from `test_sft`

### 3.3 Single-turn conversion
To keep the project simple and stable, each multi-turn dialogue is converted into a **single-turn instruction–response pair**:

- retain the **first user message**
- retain the **first assistant response**
- discard the remaining turns

This makes response-only supervision and token-level KD easier to implement and interpret.

### 3.4 Prompt construction
A light concise-answer instruction is inserted into the system prompt to discourage overly long outputs:

> Please answer clearly and concisely. Keep the response brief and focused, and avoid unnecessary details unless needed.

The final prompt format is:

```text
<|im_start|>system
Please answer clearly and concisely. Keep the response brief and focused, and avoid unnecessary details unless needed.
<|im_end|>
<|im_start|>user
{user_text}
<|im_end|>
<|im_start|>assistant

The target is the original first assistant response.

### 3.5 Length filtering

We keep only examples satisfying all of the following constraints:
	•	response length between 16 and 256 tokens
	•	prompt length no more than 512 tokens
	•	total length no more than 768 tokens

These filters are applied using the student tokenizer, because the training pipeline is constrained by student-side sequence length.

### 3.6 Final retained data

After filtering:
	•	36,973 training examples are retained from the 100,000 sampled candidates
	•	the retained training ratio is therefore 36.97%
	•	a filtered evaluation pool is also produced, and the main experiments use 1,000 held-out evaluation examples

Average retained lengths

On the filtered training data:
	•	average prompt length: 177.69 tokens
	•	average target length: 131.67 tokens
	•	average total length: 309.36 tokens

This produces a medium-length instruction dataset suitable for response-only SFT and token-level KD.

## 4. Training Setup

### 4.1 Response-only supervision

For all training settings, loss is computed only on assistant response tokens.
	•	prompt tokens are masked with -100
	•	only target tokens contribute to CE or KD loss

This keeps the setup consistent across SFT, vanilla KD, and UKD.

### 4.2 Student fine-tuning

The student model is fine-tuned with LoRA, using:
	•	rank (r = 16)
	•	LoRA alpha (= 32)
	•	dropout (= 0.05)

Target modules include:
	•	q_proj
	•	k_proj
	•	v_proj
	•	o_proj
	•	gate_proj
	•	up_proj
	•	down_proj

### 4.3 General training settings
	•	max sequence length: 768
	•	learning rate: 2e-5
	•	weight decay: 0
	•	gradient accumulation: 16
	•	epochs: 1

The training objective differs only in how the KD term is defined.

## 5. Methods

### 5.1 SFT baseline

Let the student distribution at response position (t) be (p_t^S), and the gold token be (y_t).
The response-only supervised loss is:

[
\mathcal{L}_{CE} = \sum_t \mathrm{CE}(y_t, p_t^S)
]

This is the standard SFT baseline and does not use the teacher during training.


### 5.2 Vanilla token-level KD

Let teacher and student logits at position (t) be (z_t^T) and (z_t^S).
With temperature (\tau):

[
p_t^T = \mathrm{softmax}(z_t^T / \tau), \qquad
p_t^S = \mathrm{softmax}(z_t^S / \tau)
]

The token-level KD loss is:

[
\mathcal{L}_{KD} = \tau^2 \sum_t \mathrm{KL}(p_t^T ,|, p_t^S)
]

The final objective is:

[
\mathcal{L} = (1-\lambda)\mathcal{L}{CE} + \lambda\mathcal{L}{KD}
]

Only response tokens are distilled.

Practical note

When teacher and student output vocabularies differ slightly, KD is computed over the shared vocabulary slice, while CE is still computed over the full student vocabulary.


### 5.3 Uncertainty-aware KD (UKD)

The motivation behind UKD is:
	•	if the teacher is confident at a token, that token should receive stronger distillation;
	•	if the teacher is uncertain, that token should receive weaker distillation.

Teacher uncertainty

Teacher uncertainty is measured using token-level entropy:

[
u_t = H(p_t^T) = -\sum_i p_{t,i}^T \log p_{t,i}^T
]

Token-wise weight

The entropy is converted into a weight:

[
w_t = \exp(-\alpha u_t)
]

Thus:
	•	low teacher entropy (\rightarrow) large weight
	•	high teacher entropy (\rightarrow) small weight

Uncertainty-aware KD loss

The weighted KD loss is:

[
\mathcal{L}_{UKD}

\tau^2
\frac{\sum_t w_t \cdot \mathrm{KL}(p_t^T ,|, p_t^S)}
{\sum_t w_t}
]

The final objective is:

[
\mathcal{L} = (1-\lambda)\mathcal{L}{CE} + \lambda\mathcal{L}{UKD}
]

So uncertainty is not introduced as a separate supervision target; instead, it directly modulates the strength of token-level distillation.

## 6. Evaluation Protocol

All models are evaluated on the same held-out subset of 1,000 examples.

### 6.1 Generation metrics

We compute:
	•	Exact Match
	•	Normalized Exact Match
	•	Token Overlap F1
	•	Average generated length

Among these, Token Overlap F1 is the most informative generation metric in this project, since exact match is extremely strict for open-ended instruction responses.

### 6.2 Teacher-forcing metrics

We also evaluate student fit to the gold response under teacher forcing:
	•	response-only NLL
	•	response-only PPL
	•	student mean confidence

These metrics are useful because they capture how well the student models the gold target, which is not always reflected by lexical-overlap scores alone.

### 6.3 Teacher-entropy bucket analysis

At evaluation time, the same teacher model is used to compute the mean token entropy on each gold response.

Examples are split into:
	•	low-entropy bucket
	•	mid-entropy bucket
	•	high-entropy bucket

This bucket analysis is used to answer:

On examples where the teacher itself is uncertain, does UKD behave more robustly than vanilla KD?

Importantly, for the SFT baseline, the teacher is not used during training.
It is used only at evaluation time as a common difficulty estimator.

## 7. Main Results

All comparisons below are based on the 1,000-example evaluation set.


### 7.1 SFT baseline

SFT serves as the main supervised reference point.

Its main characteristics are:
	•	strongest gold-target fitting behavior,
	•	lowest response NLL/PPL among the three settings,
	•	but weaker lexical-overlap performance than both KD variants.

Teacher-entropy bucket analysis shows a clear difficulty pattern:
	•	performance is strongest on low-entropy examples,
	•	weaker on mid-entropy examples,
	•	and worst on high-entropy examples.

This confirms that teacher entropy is a meaningful difficulty signal for this task.


### 7.2 Vanilla KD vs SFT

Compared with the SFT baseline, vanilla KD shows:
	•	+7.53% relative improvement in overall token-overlap F1
	•	exact match rises from 0.00% to 0.70%
	•	but response NLL becomes 5.32% worse
	•	and response PPL becomes 5.48% worse

By entropy bucket

Low-entropy bucket
Compared with SFT:
	•	F1 improves by 19.16%
	•	NLL worsens by 8.28%

Mid-entropy bucket
Compared with SFT:
	•	F1 improves by 2.10%
	•	NLL worsens by 5.90%

High-entropy bucket
Compared with SFT:
	•	F1 decreases by 0.89%
	•	NLL worsens by 3.83%

Interpretation

Vanilla KD is most effective on examples where the teacher is confident.
It substantially improves lexical overlap on easier examples, but its benefit weakens or disappears on high-uncertainty examples.


7.3 UKD vs SFT

Compared with the SFT baseline, UKD shows:
	•	+6.62% relative improvement in overall token-overlap F1
	•	exact match rises from 0.00% to 0.40%
	•	response NLL becomes 4.63% worse
	•	response PPL becomes 4.82% worse

By entropy bucket

Low-entropy bucket
Compared with SFT:
	•	F1 improves by 15.42%
	•	NLL worsens by 7.37%

Mid-entropy bucket
Compared with SFT:
	•	F1 improves by 2.33%
	•	NLL worsens by 5.17%

High-entropy bucket
Compared with SFT:
	•	F1 improves by 0.47%
	•	NLL worsens by 3.24%

Interpretation

UKD still improves lexical-overlap generation quality relative to SFT, but does so more conservatively than vanilla KD.
Its main benefit is not maximum overall gain, but slightly better robustness under uncertainty.


### 7.4 UKD vs Vanilla KD

This is the most important comparison.

Relative to vanilla KD, UKD shows:
	•	overall F1 is only 0.84% lower
	•	but overall response NLL is 0.66% better
	•	and overall response PPL is 0.63% better

High-entropy bucket

On the most difficult examples, UKD is slightly more robust than vanilla KD:
	•	F1 is 1.36% higher
	•	response NLL is 0.56% better
	•	response PPL is also slightly better

Interpretation

Vanilla KD achieves the strongest overall lexical-overlap score, but UKD slightly reduces the degradation observed on high-uncertainty examples.
This is exactly the intended role of uncertainty-aware weighting.


## 8. Final Takeaways

The three training settings have different strengths.

### SFT
	•	best gold-target fit,
	•	lowest NLL/PPL,
	•	weakest lexical-overlap performance.

### Vanilla KD
	•	best overall lexical-overlap performance,
	•	strongest gains on low-uncertainty examples,
	•	but weaker robustness on high-uncertainty examples.

### UKD
	•	preserves most of the lexical-overlap gains of vanilla KD,
	•	slightly improves robustness on difficult examples,
	•	and best matches the intended method design.


## 9. Conclusion

This project shows that:
	1.	token-level KD improves instruction-generation overlap relative to pure SFT,
	2.	vanilla KD works best when the teacher is confident,
	3.	high teacher uncertainty is a meaningful signal for harder examples,
	4.	entropy-based UKD can slightly reduce degradation on those harder examples,
	5.	even a simple uncertainty-aware weighting mechanism is enough to produce a more robust KD variant.

In short:

Vanilla KD gives the strongest overall lexical-overlap gain, while UKD offers a better trade-off between distillation benefit and robustness under teacher uncertainty.

⸻

## 10. Reproduction Workflow

The repository can be run in the following order:
	1.	preprocess UltraChat into single-turn prompt–response pairs
	2.	train the response-only SFT baseline
	3.	train the vanilla KD model
	4.	train the uncertainty-aware KD model
	5.	run unified evaluation on the held-out 1k subset
	6.	compare overall metrics and entropy-bucketed metrics