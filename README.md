# Filling the Gap

**An internal signature of inference about groups without evidence, and how to intervene on it without modifying weights.**

*Qwen 2.5 7B Instruct · BBQ · September 2026*

When a language model is asked about people and the context does not contain the information needed to answer, it can either admit that the answer cannot be known, or **fill the gap with a prior about a social group**. This project asks whether that operation leaves a clear signature in the model's activations, and whether that signature can be used to change the behavior **without retraining or modifying a single weight**.

The full write-up is in [`articulo_en.docx`](articulo_en.docx). The detailed, step-by-step experiment log (in Spanish) is in [`reporte.md`](reporte.md).

---

## Key findings

- **The model's main failure is not leaning toward stereotypes, but guessing when it should not.** On ambiguous BBQ questions, it chose a group in 1008 items: 502 times the stereotyped group and 506 times the anti-stereotyped one. An aggregate directional bias score ($s_{amb} = -0.004$) would rate it almost perfect.
- **Filling the gap and choosing whom are two distinct objects.** Their internal directions are nearly orthogonal ($\cos = 0.232$).
- **A signal that reads well is not necessarily useful to change behavior.** The direction separating "answers" from "abstains" reaches a holdout AUC of 0.946, but pushing on it only produces indiscriminate abstention: accuracy on `disambig` drops 33 points.
- **The outcome of reasoning is anticipated before reasoning starts.** A direction separates the cases in which the model will abandon a correct answer during its reasoning from those in which it will hold it. It is readable before the model writes a single word (paired AUC up to 0.759), with the same direction as after the first sentence ($\cos = 0.959$ at layer 22). This holds for *abandoning* a correct answer, not for *recovering* from a wrong one.
- **Subtracting that direction during generation raises accuracy by +0.096 to +0.150** on the items where the model starts on the correct answer and could abandon it, with no significant loss on `disambig` at layer 22. It fixes errors toward both groups in the same proportion (18 stereotyped and 18 anti-stereotyped in one run).
- **The layer choice remains open.** Reading quality, effect magnitude and specificity against random directions point to different layers. Earlier layers increase both the effect and the collateral damage.
- **Limits.** The effect lives in 1.98% of the ambiguous items and is not detectable in the general population. The direction reads unseen categories but does not transfer when written. As a detector, the probe is modest (lift 1.2–1.4; 1.87 in the top 10% before reasoning).

---

## Motivation and related work

Traceability matters as AI takes part in more everyday decisions, and the reasoning a model writes does not always reflect how it reached its answer. This work looks for that logic inside the model instead of in its text.

| work | what we took from it |
|---|---|
| **Chen et al. (2025)**, *Persona Vectors* | Main inspiration. Same model and same additive intervention. Their prediction from the last prompt token works mainly for explicit prompt differences; filling the gap is a subtle change, which is exactly the open case. |
| **Højer et al. (2025)**, *Representation Engineering* | The method: reading at the last token, three ways of building a vector, and one vector that pushes in both directions depending on its sign. Here the contrast comes from the model's own behavior. |
| **Dherin et al. (2025)**, *Implicit dynamics of in-context learning* | Theoretical support for working without weights: context is functionally equivalent to a small weight update whose vector part resembles a steering vector. |
| **Zhou et al. (2026)**, *The Geometry of Reasoning* | The natural time axis is the reasoning step, which led us to read the answer after every sentence. Separating logic from surface words motivates our paired items. |
| **Helff et al. (2026)**, *ActivationReasoning* | Projections with thresholds as named concepts and logical rules, the kind of traceable guardrail that motivates this work. |

The interest goes beyond bias. A system optimized only on its final answer can reach it through any process, which is the shape of the *reward hacking* problem. If the internal state contains information about the outcome before it happens, it can be read and acted on.

---

## Setup

- **Model.** `Qwen/Qwen2.5-7B-Instruct` in `bfloat16`, unquantized, on a 24 GiB GPU. All inference under `torch.no_grad()`. No weight is modified; this is verified with a SHA-256 fingerprint of parameter tensors before and after every intervention sweep.
- **Dataset.** [BBQ](https://huggingface.co/datasets/oskarvanderwal/bbq) (*Bias Benchmark for QA*). Each item has a context, a question and three options (stereotyped, anti-stereotyped, abstention), in two conditions:
  - `ambig`: the context lacks the information, so the correct answer is always to abstain;
  - `disambig`: the context contains the answer; used as a control.
- **Categories.** Eight categories for fitting and evaluation (Age, Disability_status, Nationality, Physical_appearance, Race_ethnicity, Race_x_SES, Religion, SES). Three held out from the start (Gender_identity, Race_x_gender, Sexual_orientation).
- **Reading an answer.** No text is generated for scoring. One forward pass is run and the answer is the letter (A, B or C) with the highest next-token log-probability, mapped to its role through BBQ's annotations. Each item is shown in three option orders and the majority role is taken.
- **Activations.** Always taken at the last token of the input, $a^{(\ell)} \in \mathbb{R}^{3584}$, for layers $\ell = 0 \dots 28$.

---

## Experiment 1 — the direct answer

**Question.** Can the decision to fill the gap be read in the model's state when it reads the question?

**Method.**
1. Measure how often the model fills the gap ($\rho_{unk}$) and toward which group ($s_{amb}$).
2. Contrast `ambig` items where the model chooses a group against items where it abstains, matched 1:1 by category, template, polarity and question text (1008 pairs).
3. Build three directions (reading, paired contrastive, PCA), choose the layer on a selection split and evaluate once on a holdout.
4. Intervene with $h^{(\ell)} \leftarrow h^{(\ell)} + \alpha v$, against five random directions of equal norm. Repeat for the stereotyped-vs-anti-stereotyped direction.

**Results.**

| direction or baseline | holdout AUC |
|---|---|
| **paired contrastive, layer 21** | **0.946** (95% CI 0.926–0.960) |
| TF-IDF lexical classifier | 0.887 |
| PCA, layer 19 | 0.885 |
| reading, layer 20 | 0.805 |
| layer-0 embeddings | 0.754 |

| at $\alpha = -0.5$ | `ambig` | `disambig` |
|---|---|---|
| no intervention | 0.983 | 0.785 |
| direction of the operation | 1.000 | **0.455** |
| random directions, same dose | 0.975 | 0.778 |

The direction distinguishes "will answer" from "will abstain", but pushing it withdraws the answer everywhere. The stereotype direction could not even be read (AUC 0.579, $p = 0.10$).

---

## Experiment 2 — reasoning as the object of measurement

**Question.** Can the outcome of the model's reasoning be read before it happens, and can it be changed?

**Method.**
1. Take the 1008 `ambig` items where the model, answering directly, chose a group, and generate one greedy reasoning trace per item (*"Think step by step about what the context does and does not tell us."*).
2. After each sentence, read the answer again, which gives a **role trajectory** per item.
3. Group items by their role after the first sentence (the **anchor**) and compare outcomes within an anchor, matched by category, template and polarity.
4. Build a paired direction per layer and measure its AUC with pair-grouped cross-validation (5 folds × 50 repetitions, within-pair permutations, Bonferroni over 10 layers).
5. Intervene while the model generates, with random directions of equal norm and a `disambig` control, using exact McNemar tests.
6. Test transfer to an unseen category (Age), the general population, and use as a detector.

**What reasoning does.** Reasoning changes the answer in a third of the cases, and accuracy goes from 30.5% after the first sentence to 54.6% at the end.

| role after the first sentence | items | ends in `unknown` | ends in stereotyped | ends in anti-stereotyped |
|---|---|---|---|---|
| `unknown` | 307 | 220 (71.7%) | 46 (15.0%) | 41 (13.4%) |
| `stereotyped` | 363 | 178 (49.0%) | 178 (49.0%) | 7 (1.9%) |
| `anti_stereotyped` | 338 | 152 (45.0%) | 4 (1.2%) | 182 (53.8%) |

**The signature.** In the group anchors, unpaired AUCs (0.64–0.80) disappear when pairing, so they were the item's topic. In the `unknown` anchor (31 pairs), the signal survives:

| layer | 5 | 10 | 15 | 18 | 20 | 21 | 22 | 24 | 26 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|
| after the first sentence | .564 | .547 | .608 | .657 | **.732** | **.743** | **.739** | .724 | .726 | .723 |
| before reasoning | .571 | .593 | .649 | .666 | .721 | .759 | **.759** | .752 | .734 | .726 |

**Intervention** (layer 22, 200 items):

| arm | accuracy `ambig` | Δ | p | fixed / broken | Δ `disambig` | p |
|---|---|---|---|---|---|---|
| no intervention | 0.725 | — | — | — | — | — |
| $\alpha_{eff} = -0.125$ | 0.805 | +0.080 | 0.0052 | 23 / 7 | +0.008 | 1.00 |
| $\alpha_{eff} = -0.25$ | **0.875** | **+0.150** | **<0.0001** | **36 / 6** | −0.017 | 0.79 |

A second run on 125 items gives +0.096 ($p = 0.017$, $z = 4.9$ against random directions). The smaller dose does not replicate.

**Layer profile** (125 test items, dose −0.25):

| layer | Δ `ambig` | z vs random | Δ `disambig` |
|---|---|---|---|
| 18 | +0.216 | 2.2 | −0.213 |
| 20 | +0.192 | 5.0 | −0.113 |
| 22 | +0.096 | 4.9 | −0.038 |

**How far it reaches.**
- **Reading transfers.** Built without Age it reads Age (AUC 0.653); built only with Age it reads the other categories (0.679).
- **Writing does not.** Added on 130 Age items, it leaves accuracy unchanged (17 fixed, 16 broken).
- **General population.** On 220 arbitrary ambiguous items, the model is already right on 95.5%. The direction gives +0.036 and a random direction +0.032, so the effect cannot be attributed to the direction.
- **Detector.** Precision is 1.2–1.4 times the 32.7% base rate. Read before reasoning, the top 10% by score abandon 61.1% of the time.

---

## Discussion

- **Filling the gap and choosing whom are two distinct objects.** The intervention reduces the volume of error with no group preference. The few errors it introduces lean toward the stereotyped group (17 against 8, not significant), which is worth watching at higher doses.
- **Reasoning carries out a prior disposition.** The disposition to abandon the correct answer is present in the model's state before reasoning, and reasoning partly unfolds it. The signal is at chance at layers 5–15, so it is a computation on the question, not a surface feature of the text.
- **Reading well does not guarantee a good signal for intervention.** What matters is what the direction contrasts and where it is written.
- **Detecting is not the same as controlling.** The probe is more useful for auditing than as a gate. The gate itself was not evaluated end to end.
- **Scope and precision.** Effects are reported as ranges. The vector comes from 31 pairs, and about 10% of verdicts change across runs with different batching.

---

## Future work and conclusions

- **Choosing the intervention layer** with a criterion defined in advance and different from the metric used to judge the intervention.
- **Transfer in writing** to unseen categories. The layer, the vector and the population are all possible causes of the null result.
- **Amount of data** a steering vector needs, beyond the 31 pairs available here.
- **Per-axis sampling** to decide whether a common prior direction exists across axes.
- **Other behaviors** where the final result does not reveal how it was reached, such as reward hacking.

---

## Repository layout

```
src/bbq_gate/     scoring, role resolution, contrast building and direction construction
scripts/          experiment scripts, numbered in execution order
data/             knowledge graph: questions, experiments, decisions and references (see data/index.md)
openspec/specs/   specifications (bbq-existence-gate, control-vector)
tests/            unit tests and the knowledge-graph validator (validate_graph.py)
reporte.md        detailed experiment log, in Spanish
articulo_en.docx  full article, in English
```

| experiment | scripts |
|---|---|
| Existence gate and census | `02_bbq_existence_gate.py`, `03_bbq_category_sweep.py`, `05_recompute_gate.py` |
| Experiment 1 | `06_build_operation_vector.py`, `07_steering_h2.py`, `08_direction_vector_h3.py` |
| Exploration of reasoning | `09` to `14` |
| Experiment 2, signature | `15_revision_signature_scaled.py`, `17_abandonment_signature.py`, `19_recovery_signature.py`, `20_paired_fold_correction.py`, `21_step0_baseline.py`, `24_repeated_cv.py` |
| Experiment 2, intervention | `16_revision_steering.py`, `18_steering_confirmation.py`, `25_layer_sweep.py`, `26_specificity_profile.py`, `26b_profile_recompute.py` |
| Experiment 2, transfer and checks | `22_category_transfer.py`, `23_causal_transfer_age.py`, `26c_overlap_reproducibility.py`, `26d_symmetry_consistency.py` |

Results and their interpretation are recorded in `data/experiments/EXP-002_bbq_stereotype_direction/results.md`.

---

## Reproducing

The environment runs on WSL (Ubuntu 22.04) with a Python virtual environment on ext4 and a CUDA GPU with at least 24 GiB.

```bash
bash scripts/setup_wsl_env.sh     # creates the environment from requirements.txt
python -m pytest -q                # unit tests
python tests/validate_graph.py --strict
```

Run the BBQ scripts in numerical order, from `02` onward (`00`, `01` and `04` belong to an earlier exploration on HatEval and HateCheck). Greedy generation is deterministic only when batch composition is kept, so compare arms within the same run.

---

## References

- Chen, Arditi, Sleight, Evans, Lindsey (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*. arXiv:2507.21509.
- Højer, Jarvis, Heinrich (2025). *Improving Reasoning Performance in Large Language Models via Representation Engineering*. ICLR 2025. arXiv:2504.19483.
- Dherin, Munn, Mazzawi, Wunder, Gonzalvo (2025). *Learning without training: The implicit dynamics of in-context learning*. arXiv:2507.16003.
- Zhou, Wang, Yin, Zhou, Zhang (2026). *The Geometry of Reasoning: Flowing Logics in Representation Space*. ICLR 2026.
- Helff, Härle, Stammer, Friedrich, Brack, Wüst, Shindo, Schramowski, Kersting (2026). *ActivationReasoning: Logical Reasoning in Latent Activation Spaces*. ICLR 2026.
- Parrish et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question Answering*. Findings of ACL 2022.
