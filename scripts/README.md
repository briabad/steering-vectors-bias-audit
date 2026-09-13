# `scripts/` — index

Every measurement in this project is produced by one of these scripts. They are numbered roughly in
the order they were written, which is **not** a pipeline: several exist to correct an earlier one,
and the later ones re-analyse data that is already on disk.

For what the results mean, see the top-level [README](../README.md), the article
(`articulo_en.docx`) and the full log in `reporte.md`.

## Conventions

- **Answers are read, not generated.** One forward pass, and the answer is the letter (A, B or C)
  with the highest next-token log-probability, mapped to its role. Free text is never parsed.
- **No weight is ever modified.** Interventions are forward hooks on the residual stream.
- **Outputs** are written to `data/experiments/EXP-00X_.../output/`, one JSON report per script,
  plus `.jsonl` per-item records and `.npz` activations where relevant.
- **Long runs write partial state** so they can be resumed, and random seeds are fixed in the file.
- **The model** is `Qwen/Qwen2.5-7B-Instruct` in bf16 on a 24 GiB GPU; GPU scripts are marked below.

---

## Environment and workspace checks

| script | what it does |
|---|---|
| `setup_wsl_env.sh` | Builds the virtual environment inside WSL on ext4 (`~/.venvs/niel_landa`), not on `/mnt/c`. |
| `check_disk_space.sh` | Checks free space on the ext4 filesystem before downloading model weights. |
| `validate_setup.py` | Verifies that the project structure is in place. |
| `00_validate_implementation.py` | Checks the `src/bbq_gate` implementation without a GPU: imports, domain logic, integration and output format. |
| `run_full_validation.sh` | Runs tests, the knowledge-graph check and a real gate execution end to end. |

---

## Earlier line — hate speech (EXP-001)

This line was superseded by the BBQ work, but it produced the frozen baselines the rest is judged
against. Outputs go to `EXP-001_hateval_vector_flow/output/`.

| script | what it does | output |
|---|---|---|
| `00_lexical_baseline.py` | The frozen lexical yardstick. TF-IDF plus logistic regression on HatEval and HateCheck, with no model at all. Establishes how much "capability" is explained without any semantic computation. | `lexical_baseline.json` |
| `01_hateval_latent_pipeline.py` | First proof-of-concept pipeline on HatEval: contrastive subset, activations, mean-difference vector and geometry. Has three known defects and is pending repair or removal (FEAT-019). **GPU** | `dataset_summary.json`, `flow_metrics_reference.json` (not currently on disk) |
| `04_flow_geometry_probe.py` | Probe on HateCheck minimal pairs: does per-token flow geometry separate sentences with the same words and different logical structure? Negative result. **GPU** | `flow_geometry_probe.json` |

---

## Existence gate and census (BBQ)

From here on, outputs go to `EXP-002_bbq_stereotype_direction/output/`.

| script | what it does | output |
|---|---|---|
| `02_bbq_existence_gate.py` | The main evaluation: 11 BBQ categories, 3 option orders, 2 prompt formats, ~288,000 forward passes (~2 h on an RTX 4090). Persists one raw record per observation. **GPU** | `existence_gate_raw.jsonl`, `existence_gate.json` |
| `03_bbq_category_sweep.py` | Cheap exploratory sweep over the 11 categories (150 items each) that decided the scope. **GPU** | `category_sweep.json` |
| `05_recompute_gate.py` | Recomputes the gate from the raw JSONL with both criteria, consolidating by majority of permutations at the item level. Exists because the aggregation in `02` produced NaN over a valid run; no GPU needed. | `gate_recomputed.json` |

---

## Experiment 1 — the direct answer

| script | what it does | output |
|---|---|---|
| `06_build_operation_vector.py` | Builds and evaluates $v_{op}$: reconstructs item identity from the raw corpus, pairs 1:1 by category, template and polarity, extracts activations in one pass, and compares the three constructions against the lexical and layer-0 baselines. **GPU** | `operation_vector.json`, `operation_vector_activations.npz` |
| `07_steering_h2.py` | Causal test of $v_{op}$ (H2): sweeps $\alpha$ with baseline, direction and five random directions of equal norm, on items outside the construction. **GPU** | `steering_h2.json` |
| `08_direction_vector_h3.py` | Builds $v_{dir}$, the stereotyped-versus-anti-stereotyped direction, and tests it causally in the same file (H3). Also measures $\cos(v_{dir}, v_{op})$. **GPU** | `direction_vector_h3.json` |

---

## Exploring reasoning

These scripts chart the terrain that Experiment 2 then measured properly. Each one answers a
question raised by the previous.

| script | what it does | output |
|---|---|---|
| `09_meanpool_and_cot_faithfulness.py` | Two checks: whether mean pooling over the prompt reads the group choice better than the last token, and whether generated reasoning changes the answer at all. **GPU** | `meanpool_cot.json` |
| `10_cot_yield_pilot.py` | Sampling pilot: does temperature produce divergent traces for the same item, enough for a within-item contrast? **GPU** | `cot_yield_pilot.json` |
| `11_reasoning_step_direction.py` | Same 107 pairs as the prompt-based contrast, but extracting per reasoning step instead of from the prompt. Only the extraction point changes. **GPU** | `reasoning_step_direction.json` |
| `12_deliberation_contrast.py` | Contrast by what happened to the process: reasoning that changed the answer versus reasoning that did not. **GPU** | `deliberation_contrast.json` |
| `13_change_direction_and_nonlinear.py` | Adds per-item records (the lesson from losing aggregates in `12`), splits changes by direction, and tries a non-linear probe. **GPU** | `change_direction_nonlinear.json`, `deliberation_records.jsonl`, `deliberation_step1_acts.npz` |
| `14_revision_signature.py` | Fixes the tautology in `13`: the label was determined by the very variable used to predict it. Fixes the role at step 1 and varies only the outcome. **GPU** | `revision_signature.json` |

---

## Experiment 2 — reasoning as the object of measurement

### The signature

| script | what it does | output |
|---|---|---|
| `15_revision_signature_scaled.py` | Scales the contrast from 214 to 1008 items: generates one full reasoning trace per item, reads the answer after every sentence, and stores the complete role trajectory and the activations at 11 layers. **GPU** | `revision_scaled.json`, `revision_scaled_records.jsonl`, `revision_scaled_acts.npz` |
| `17_abandonment_signature.py` | Relabels the `unknown` anchor: abandons (87) against holds (146), discarding the 74 that oscillate. This is the contrast that yields the signature. | `abandonment_signature.json` |
| `19_recovery_signature.py` | The mirror population: items that are already on a wrong answer after the first sentence, and the question of who recovers. | `recovery_signature.json` |
| `20_paired_fold_correction.py` | Corrects fold assignment in the paired contrasts of `15`, `17` and `19`: splitting a pair across folds biases the AUC downward. | `paired_fold_correction.json` |
| `21_step0_baseline.py` | Extracts the same activations **before any reasoning**, in two prompt variants, to decide whether the signal is computed during the reasoning or was already in the question. **GPU** | `step0_baseline.json`, `step0_acts.npz` |
| `24_repeated_cv.py` | Stabilises the AUCs with repeated cross-validation after finding up to 6 points of variation between single fold draws. | `repeated_cv.json` |

### The intervention

| script | what it does | output |
|---|---|---|
| `16_revision_steering.py` | First causal test of $v_{aband}$: dose calibration and dose curve at layer 22, with random arms. Aggregates only. **GPU** | `revision_steering.json`, `revision_steering_degenerate_alpha.json` |
| `18_steering_confirmation.py` | Paired confirmation with per-item records, so McNemar applies. `--population fresh` runs the same intervention on arbitrary ambiguous items instead of the anchor. **GPU** | `steering_confirmation.json`, `steering_confirmation_items.jsonl` (`--suffix` for variants, e.g. `_fresh`) |
| `25_layer_sweep.py` | Causal layer sweep with a registered development/test split written to disk before any measurement. **GPU** | `layer_sweep.json`, `layer_sweep_split.json`, `layer_sweep_items.jsonl` |
| `26_specificity_profile.py` | Full profile on the test split: magnitude, collateral damage and specificity against random directions, per layer and dose. **GPU** | `specificity_profile.json`, `specificity_profile_items.jsonl` |
| `26b_profile_recompute.py` | Recomputes the profile using every available random seed, and reports mean and spread of the random arm instead of just the maximum. | `specificity_profile_fixed.json` |

### Transfer and consistency checks

| script | what it does | output |
|---|---|---|
| `22_category_transfer.py` | Leave-one-category-out in **reading**: a direction built without Age evaluated on Age, and the reverse. Transfer and confounder control in one test. | `category_transfer.json` |
| `23_causal_transfer_age.py` | The same in **writing**: the vector built from the 15 non-Age pairs, applied to 130 Age items. Disjointness holds by construction. **GPU** | `causal_transfer_age.json`, `causal_transfer_age_items.jsonl` |
| `26c_overlap_reproducibility.py` | Reconstructs which items each evaluation saw and compares them one by one, to tell sampling apart from non-determinism in greedy generation. | prints to stdout |
| `26d_symmetry_consistency.py` | Counts, across runs, whether the symmetry of corrected errors holds. Generates nothing, only counts. | prints to stdout |
| `27_letter_mass_check.py` | Checks the reading instrument itself: how much probability mass the three letters hold, and whether the most likely token over the whole vocabulary is one of them. Four conditions (before reasoning, after the first sentence, direct answer in chat and in plain format), so a constant bias can be told apart from a differential one. **GPU** | `letter_mass_check.json`, `letter_mass_check_items.jsonl` |

---

## Notes

- **Not everything lives in a script.** The detector operating curve (recall, precision and lift by
  percentile) was computed as an ad-hoc analysis over the probe scores, and is reported in
  `results.md` and in the article, but is not one of the scripts above.
- **Corrections are kept, not overwritten.** `05` corrects `02`, `14` corrects `13`, `20` corrects
  `15`/`17`/`19`, `24` corrects single-draw AUCs, `26b` corrects `26`. The superseded run stays on
  disk so the history is auditable.
- **Some scripts sit outside the OpenSpec flow** by explicit decision, starting with `07` and `08`.
  They are measurement scripts, not application code.
