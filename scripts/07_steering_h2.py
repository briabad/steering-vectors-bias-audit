#!/usr/bin/env python3
"""H2 causal steering experiment (EXP-002, hypothesis.md Addendum 8).

Measurement script, deliberately OUTSIDE the OpenSpec flow -- Addendum 8's
"Nota de proceso": "Este experimento se implementa como script de medición en
scripts/, al modo de 00_lexical_baseline.py y 05_recompute_gate.py, fuera del
flujo de OpenSpec." Authorized deviation from CLAUDE.md, by explicit user
decision, to prioritize the result.

Intervention: x_22 += alpha * v, at the residual stream OUTPUT of decoder
layer 21 (1-indexed, i.e. hidden_states[21] == the output of
`model.model.layers[20]` in this transformers version, where
Qwen2DecoderLayer.forward returns a plain tensor -- verified against the
installed transformers==4.57.6 source). v_op points from "abstained" toward
"guessed" (built in `scripts/06_build_operation_vector.py`), so alpha < 0
suppresses the guess.

Three arms, all mandatory (Addendum 8):
  1. alpha = 0             -- no intervention (shared baseline for both arms)
  2. alpha * v_op           -- the operation direction (primary_direction,
                               layer 21, contrastive construction, from
                               `output/operation_vector.json`)
  3. alpha * r, ||r||=||v_op||  -- >= 5 random directions, norm-matched,
                               registered seeds

Evaluation set: BBQ items from the 8 ADJUSTMENT categories (the 3 reserved
categories -- Gender_identity, Race_x_gender, Sexual_orientation -- stay
untouched, per Addendum 5/8, for H3), ambig AND disambig, EXCLUDING every
item whose activations were used anywhere in H1's fitting/freeze/holdout
pipeline (the full 1008-pair, 2016-item fused working set persisted in
`operation_vector_activations.npz`'s `keys`).

Same scoring method as the existence gate (`infrastructure/scorer.py`):
log-likelihood of the letter token ("A"/"B"/"C", max over the bare and
space-prefixed variant) at the last token position, chat format, canonical
(unpermuted) option order -- matching the exact context v_op was built in
(format='chat', permutation=[0,1,2] in `06_build_operation_vector.py`), so
alpha is directly comparable to the H1 pipeline. This is a deliberate,
documented narrowing of "misma puntuación... que la puerta de existencia"
(which itself averaged 2 formats x 3 permutations): replicating that full
6x factor here would multiply the already-large steering sweep's cost
six-fold for a diagnostic axis (format/permutation robustness) that is not
part of C-H2a/b/c.

KL divergence / entropy / probability-of-correct-answer are computed over
the FULL vocabulary next-token distribution (SOA-004 eq. 5-8: "divergencia
KL entre la distribucion de logits con y sin intervencion"), not a
renormalized 3-way distribution, to stay faithful to the cited method.

Usage:
    python scripts/07_steering_h2.py [-v] [--smoke]

--smoke runs a tiny configuration (2 items/category/condition, 2 alphas,
still 5 random seeds) against `_smoke/steering_h2.json`, to validate the
mechanics without spending real GPU time on the canonical path.

Runtime note: see `estimate_and_choose_alpha_step` -- a timing pilot (the
real alpha=0 pass, timed) is used to project the full sweep's cost and
decide between step=0.25 (145 configs) and the Addendum-8-compliant
fallback step=0.5 (73 configs, same >=5 random directions) BEFORE running
it, per the leader's instruction ("si va a pasar de una hora, reduce el
paso de alpha a 0.5... no reduzcas el numero de direcciones aleatorias").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bbq_gate.application.vector_construction import item_key
from bbq_gate.domain.contrast import RESERVED_CATEGORIES
from bbq_gate.infrastructure.activation_store import load_activations
from bbq_gate.infrastructure.loaders import BBQ_CATEGORIES, load_bbq_category
from bbq_gate.infrastructure.prompts import build_prompt

# --------------------------------------------------------------------------
# Constants (Addendum 8, fixed BEFORE running any intervention)
# --------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_STATE_LAYER = 21  # matches operation_vector.json's h1_evaluation.primary_layer
MODULE_LAYER_INDEX = HIDDEN_STATE_LAYER - 1  # model.model.layers[20] output == hidden_states[21]
FORMAT = "chat"
CANONICAL_PERMUTATION = [0, 1, 2]
LETTERS = ["A", "B", "C"]

ALPHA_MIN, ALPHA_MAX = -3.0, 3.0
ALPHA_STEP_DEFAULT = 0.25
ALPHA_STEP_FALLBACK = 0.5
TIME_BUDGET_SECONDS = 3600.0

N_RANDOM_DIRECTIONS = 5
RANDOM_SEEDS = [20260906 + i for i in range(1, N_RANDOM_DIRECTIONS + 1)]  # [20260907..20260911], registered

N_PER_CATEGORY_AMBIG = 50
N_PER_CATEGORY_DISAMBIG = 50
SAMPLE_SEED = 20260906
BOOTSTRAP_SEED = 20260906
N_BOOTSTRAP = 1000

BATCH_SIZE = 32

C_H2A_THRESHOLD = 0.10  # 10 points
C_H2B_THRESHOLD = 0.05  # 5 points

ALIAS = {"f": {"woman", "girl", "female"}, "m": {"man", "boy", "male"}}

OUTPUT_DIR = REPO / "data" / "experiments" / "EXP-002_bbq_stereotype_direction" / "output"
OPERATION_VECTOR_PATH = OUTPUT_DIR / "operation_vector.json"
ACTIVATIONS_PATH = OUTPUT_DIR / "operation_vector_activations.npz"
RESULT_PATH = OUTPUT_DIR / "steering_h2.json"
PARTIAL_PATH = OUTPUT_DIR / "steering_h2_partial.json"


def check_disk_space(min_gb: int = 3) -> bool:
    stat = shutil.disk_usage(REPO)
    free_gb = stat.free / (1024**3)
    print(f"[DISK] {free_gb:.1f} GB free (need {min_gb} GB)")
    return free_gb >= min_gb


def alpha_grid(step: float) -> list[float]:
    n = round((ALPHA_MAX - ALPHA_MIN) / step)
    return [round(ALPHA_MIN + i * step, 4) for i in range(n + 1)]


# --------------------------------------------------------------------------
# Section 1: eval set -- non-reserved categories, ambig+disambig, excluding
# every item used anywhere in H1's fitting/holdout pipeline.
# --------------------------------------------------------------------------


def load_excluded_keys() -> set[str]:
    """Every item_key present in the fused working set's persisted
    activations (`operation_vector_activations.npz`): the 1008-pair,
    2016-item union of v_A (strict-key pairs) and v_B (category-matched
    residue) used for sub_adjuste + layer_selection + holdout in H1. All
    such items are `ambig` by construction (the operation contrast only
    exists within `ambig`); `disambig` items were never used to build or
    evaluate v_op.
    """
    if not ACTIVATIONS_PATH.exists():
        raise RuntimeError(
            f"{ACTIVATIONS_PATH} not found; cannot determine which items were used "
            f"to build v_op (required to keep the H2 eval set disjoint from H1's fitting set)"
        )
    keys, _array, _metadata = load_activations(ACTIVATIONS_PATH)
    return set(keys)


def build_eval_set(
    n_per_category_ambig: int, n_per_category_disambig: int, verbose: bool
) -> tuple[list[dict], dict]:
    """Sample a stratified, held-out (never used to build v_op) eval set from
    the 8 adjustment categories, ambig + disambig.

    Returns:
        (eval_items, diagnostics): eval_items is a list of dicts with
        category/condition/item_id/item_key/correct_position/prompt_item
        (the BBQItem itself, kept for prompt construction after the
        tokenizer is loaded). diagnostics records per-category availability
        and sampling counts for the output's reproducibility section.
    """
    excluded_keys = load_excluded_keys()
    categories = [c for c in BBQ_CATEGORIES if c not in RESERVED_CATEGORIES]
    assert len(categories) == 8, f"expected 8 adjustment categories, got {len(categories)}"

    rng = random.Random(SAMPLE_SEED)
    eval_items: list[dict] = []
    per_category: dict[str, dict] = {}

    for cat in categories:
        items, _stats = load_bbq_category(cat, ALIAS, verbose=verbose)
        ambig = [it for it in items if it.context_condition == "ambig" and item_key(it) not in excluded_keys]
        disambig = [it for it in items if it.context_condition == "disambig" and item_key(it) not in excluded_keys]

        ambig_sample = rng.sample(ambig, min(n_per_category_ambig, len(ambig)))
        disambig_sample = rng.sample(disambig, min(n_per_category_disambig, len(disambig)))

        per_category[cat] = {
            "n_ambig_total": sum(1 for it in items if it.context_condition == "ambig"),
            "n_ambig_available_after_exclusion": len(ambig),
            "n_ambig_sampled": len(ambig_sample),
            "n_disambig_total": sum(1 for it in items if it.context_condition == "disambig"),
            "n_disambig_available_after_exclusion": len(disambig),
            "n_disambig_sampled": len(disambig_sample),
        }

        for it in ambig_sample:
            eval_items.append(
                {
                    "category": it.category,
                    "condition": "ambig",
                    "item_id": it.item_id,
                    "item_key": item_key(it),
                    "correct_position": it.unknown_option().position,
                    "bbq_item": it,
                }
            )
        for it in disambig_sample:
            eval_items.append(
                {
                    "category": it.category,
                    "condition": "disambig",
                    "item_id": it.item_id,
                    "item_key": item_key(it),
                    "correct_position": it.label_idx,
                    "bbq_item": it,
                }
            )

    diagnostics = {
        "n_excluded_keys_from_h1": len(excluded_keys),
        "reserved_categories_untouched": sorted(RESERVED_CATEGORIES),
        "adjustment_categories": categories,
        "per_category": per_category,
        "n_total": len(eval_items),
        "n_ambig": sum(1 for e in eval_items if e["condition"] == "ambig"),
        "n_disambig": sum(1 for e in eval_items if e["condition"] == "disambig"),
    }
    return eval_items, diagnostics


# --------------------------------------------------------------------------
# Section 2: the steering scorer -- forward hook on the decoder layer,
# letter-logit scoring identical in spirit to infrastructure/scorer.py's
# QwenScorer, plus full-vocabulary log-probs for the KL/entropy/prob-correct
# triangulation (SOA-004 eq. 5-8).
# --------------------------------------------------------------------------


class SteeringScorer:
    """Loads Qwen 2.5 7B Instruct once; a registered forward hook on
    `model.model.layers[MODULE_LAYER_INDEX]` adds `alpha * direction` to the
    residual stream at EVERY token position of that layer's output (padded
    positions are inert: `attention_mask` keeps real tokens from attending
    to them). `set_delta(None)` makes the hook a no-op, exercised for the
    alpha=0 arm and used to confirm the intervention is purely a runtime
    forward-pass effect -- no weight is ever written to.
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; refusing to run on CPU (DEC-002)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = self.model.device
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.vocab_size = int(self.model.config.vocab_size)
        self._delta: torch.Tensor | None = None
        self._letter_ids = self._build_letter_ids()
        layer = self.model.model.layers[MODULE_LAYER_INDEX]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def _build_letter_ids(self) -> dict[str, list[int]]:
        letters: dict[str, list[int]] = {}
        for letter in LETTERS:
            ids = set()
            for form in (letter, " " + letter):
                encoded = self.tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
            letters[letter] = sorted(ids)
        return letters

    def _hook_fn(self, module, inputs, output):
        if self._delta is None:
            return output
        if isinstance(output, tuple):
            hs = output[0] + self._delta.to(dtype=output[0].dtype, device=output[0].device)
            return (hs,) + output[1:]
        return output + self._delta.to(dtype=output.dtype, device=output.device)

    def set_delta(self, delta_vector: np.ndarray | None) -> None:
        """`None` disables the hook's effect (identity); otherwise `delta_vector`
        (shape (hidden_dim,), already alpha-scaled) is cast to bf16 on-device."""
        if delta_vector is None:
            self._delta = None
        else:
            self._delta = torch.tensor(np.asarray(delta_vector, dtype=np.float32), dtype=torch.bfloat16, device=self.device)

    def remove_hook(self) -> None:
        self._hook_handle.remove()

    @torch.no_grad()
    def score_batch(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (letter_scores (b,3) float32, full_log_probs (b,vocab) float32)
        at the last token position, under whatever `self._delta` is currently set."""
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        logits = self.model(**encoded).logits[:, -1, :].float()  # promote before any reduction
        log_probs = torch.log_softmax(logits, dim=-1)
        letter_scores = torch.full((log_probs.shape[0], 3), float("-inf"), dtype=torch.float32, device=log_probs.device)
        for li, letter in enumerate(LETTERS):
            ids = self._letter_ids[letter]
            if ids:
                letter_scores[:, li] = log_probs[:, ids].max(dim=1).values
        return letter_scores.cpu().numpy(), log_probs.cpu().numpy()

    def weight_fingerprint(self) -> dict[str, str]:
        """Cheap, exact integrity check: SHA-256 of a handful of parameter
        tensors' raw bytes, sampled across the model (embedding, an
        intervened-on layer, the final norm, the LM head). Comparing this
        before/after the whole sweep is the "hay que verificarlo al
        terminar" check that the hook never writes to a weight."""
        names = [
            "model.embed_tokens.weight",
            f"model.layers.{MODULE_LAYER_INDEX}.self_attn.q_proj.weight",
            f"model.layers.{MODULE_LAYER_INDEX}.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        named = dict(self.model.named_parameters())
        fp = {}
        for n in names:
            if n not in named:
                continue
            t = named[n].detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
            fp[n] = hashlib.sha256(t).hexdigest()
        return fp


# --------------------------------------------------------------------------
# Section 3: batched scoring over the whole eval set + per-item reductions
# --------------------------------------------------------------------------


def score_eval_set(
    scorer: SteeringScorer,
    prompts: list[str],
    correct_positions: np.ndarray,
    batch_size: int,
    baseline_full_log_probs: np.ndarray | None,
    keep_full_log_probs: bool,
) -> dict:
    """One full pass over `prompts` under the scorer's CURRENT delta.

    KL/entropy are computed against `baseline_full_log_probs` batch-by-batch
    (never materializing more than one batch's (b, vocab) array beyond the
    baseline's own full-eval-set store), promoting to float64 before every
    reduction (docs/conventions.md 5.1 / this repo's float32-before-reduction
    convention, applied one step further here given the wide dynamic range
    of a 150k-token vocabulary's log-probabilities).
    """
    n = len(prompts)
    chosen = np.zeros(n, dtype=np.int64)
    entropy = np.zeros(n, dtype=np.float64)
    kl = np.zeros(n, dtype=np.float64)
    prob_correct = np.zeros(n, dtype=np.float64)
    full_store = np.zeros((n, scorer.vocab_size), dtype=np.float32) if keep_full_log_probs else None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        letter_scores, log_probs = scorer.score_batch(prompts[start:end])
        log_probs_f64 = log_probs.astype(np.float64)
        probs_f64 = np.exp(log_probs_f64)
        entropy[start:end] = -np.sum(probs_f64 * log_probs_f64, axis=1)
        chosen[start:end] = letter_scores.argmax(axis=1)
        cp = correct_positions[start:end]
        prob_correct[start:end] = np.exp(letter_scores[np.arange(end - start), cp].astype(np.float64))
        if baseline_full_log_probs is not None:
            base_slice = baseline_full_log_probs[start:end].astype(np.float64)
            kl[start:end] = np.sum(probs_f64 * (log_probs_f64 - base_slice), axis=1)
        if keep_full_log_probs:
            full_store[start:end] = log_probs

    return {
        "chosen": chosen,
        "entropy": entropy,
        "kl": kl,
        "prob_correct": prob_correct,
        "full_log_probs": full_store,
    }


def reduce_metrics(result: dict, correct_positions: np.ndarray, conditions: np.ndarray) -> dict:
    correct = result["chosen"] == correct_positions
    is_ambig = conditions == "ambig"
    is_disambig = conditions == "disambig"

    def _mean(mask: np.ndarray, arr: np.ndarray) -> float:
        return float(arr[mask].mean()) if mask.any() else float("nan")

    return {
        "n_ambig": int(is_ambig.sum()),
        "n_disambig": int(is_disambig.sum()),
        "accuracy_ambig": _mean(is_ambig, correct.astype(np.float64)),
        "accuracy_disambig": _mean(is_disambig, correct.astype(np.float64)),
        "mean_entropy": float(result["entropy"].mean()),
        "mean_entropy_ambig": _mean(is_ambig, result["entropy"]),
        "mean_entropy_disambig": _mean(is_disambig, result["entropy"]),
        "mean_kl_vs_baseline": float(result["kl"].mean()),
        "mean_kl_vs_baseline_ambig": _mean(is_ambig, result["kl"]),
        "mean_kl_vs_baseline_disambig": _mean(is_disambig, result["kl"]),
        "mean_prob_correct_ambig": _mean(is_ambig, result["prob_correct"]),
        "mean_prob_correct_disambig": _mean(is_disambig, result["prob_correct"]),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(verbose: bool = False, smoke: bool = False) -> int:
    start = time.time()
    print("[INIT] EXP-002 H2 causal steering (Addendum 8)")
    if not check_disk_space():
        return 1
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return 1

    n_amb = 2 if smoke else N_PER_CATEGORY_AMBIG
    n_dis = 2 if smoke else N_PER_CATEGORY_DISAMBIG
    result_path = (OUTPUT_DIR / "_smoke" / "steering_h2.json") if smoke else RESULT_PATH
    partial_path = (OUTPUT_DIR / "_smoke" / "steering_h2_partial.json") if smoke else PARTIAL_PATH

    print("[1/7] Loading v_op (operation_vector.json)...")
    with open(OPERATION_VECTOR_PATH) as f:
        ov = json.load(f)
    primary_layer = ov["h1_evaluation"]["primary_layer"]
    if primary_layer != HIDDEN_STATE_LAYER:
        raise RuntimeError(
            f"operation_vector.json's primary_layer ({primary_layer}) != HIDDEN_STATE_LAYER "
            f"({HIDDEN_STATE_LAYER}) constant in this script; the design fixed 'capa 21' from H1's "
            f"result, and this mismatch means the source artifact changed underneath this script."
        )
    v_op = np.asarray(ov["primary_direction"], dtype=np.float64)
    v_op_norm = float(np.linalg.norm(v_op))
    print(f"      v_op: dim={v_op.shape[0]}  layer={primary_layer}  ||v_op||={v_op_norm:.4f}")

    print("[2/7] Building held-out eval set (8 adjustment categories, ambig+disambig)...")
    eval_items, eval_diag = build_eval_set(n_amb, n_dis, verbose)
    print(
        f"      n_total={eval_diag['n_total']}  ambig={eval_diag['n_ambig']}  "
        f"disambig={eval_diag['n_disambig']}  (excluded {eval_diag['n_excluded_keys_from_h1']} "
        f"items used by H1)"
    )

    print("[3/7] Loading Qwen 2.5 7B Instruct + registering the steering hook...")
    scorer = SteeringScorer()
    fingerprint_before = scorer.weight_fingerprint()

    prompts = []
    correct_positions = []
    conditions = []
    for e in eval_items:
        prompt, _ = build_prompt(e["bbq_item"], FORMAT, CANONICAL_PERMUTATION, scorer.tokenizer)
        prompts.append(prompt)
        correct_positions.append(e["correct_position"])
        conditions.append(e["condition"])
    correct_positions = np.asarray(correct_positions, dtype=np.int64)
    conditions = np.asarray(conditions)
    n_total = len(prompts)

    print("[4/7] Baseline (alpha=0) pass + timing pilot...")
    scorer.set_delta(None)
    t0 = time.time()
    baseline_result = score_eval_set(
        scorer, prompts, correct_positions, BATCH_SIZE, baseline_full_log_probs=None, keep_full_log_probs=True
    )
    baseline_elapsed = time.time() - t0
    baseline_metrics = reduce_metrics(baseline_result, correct_positions, conditions)
    items_per_sec = n_total / baseline_elapsed if baseline_elapsed > 0 else float("nan")
    print(
        f"      baseline: acc_ambig={baseline_metrics['accuracy_ambig']:.4f}  "
        f"acc_disambig={baseline_metrics['accuracy_disambig']:.4f}  "
        f"({baseline_elapsed:.1f}s, {items_per_sec:.1f} items/s)"
    )

    n_configs_step025 = 1 + 24 + N_RANDOM_DIRECTIONS * 24
    n_configs_step05 = 1 + 12 + N_RANDOM_DIRECTIONS * 12
    projected_step025 = n_configs_step025 * n_total / items_per_sec if items_per_sec > 0 else float("inf")
    projected_step05 = n_configs_step05 * n_total / items_per_sec if items_per_sec > 0 else float("inf")

    if smoke:
        alpha_step = ALPHA_STEP_DEFAULT
        step_reason = "smoke run: step fixed at 0.25 over a tiny grid, cost projection not gating"
    elif projected_step025 <= TIME_BUDGET_SECONDS:
        alpha_step = ALPHA_STEP_DEFAULT
        step_reason = (
            f"projected full sweep at step=0.25 ({n_configs_step025} configs) = "
            f"{projected_step025/60:.1f} min <= 60 min budget -> keeping step=0.25"
        )
    else:
        alpha_step = ALPHA_STEP_FALLBACK
        step_reason = (
            f"projected full sweep at step=0.25 ({n_configs_step025} configs) = "
            f"{projected_step025/60:.1f} min > 60 min budget -> falling back to step=0.5 "
            f"({n_configs_step05} configs, projected {projected_step05/60:.1f} min); "
            f"N_RANDOM_DIRECTIONS kept at {N_RANDOM_DIRECTIONS} per instructions"
        )
    print(f"[COST] {step_reason}")

    if smoke:
        alphas_all = [-0.5, 0.0, 0.5]
    else:
        alphas_all = alpha_grid(alpha_step)
    nonzero_alphas = [a for a in alphas_all if abs(a) > 1e-9]
    n_configs_actual = 1 + len(nonzero_alphas) * (1 + N_RANDOM_DIRECTIONS)
    print(f"      alpha grid: {alphas_all}")
    print(f"      total forward-pass configurations this run: {n_configs_actual}")

    rng_dirs = {}
    for seed in RANDOM_SEEDS:
        r = np.random.default_rng(seed).normal(size=v_op.shape[0])
        r_unit = r / np.linalg.norm(r)
        rng_dirs[seed] = r_unit * v_op_norm

    def build_curve_state() -> dict:
        return {
            "metadata": {
                "model_id": scorer.model_id,
                "revision": scorer.revision,
                "torch_version": torch.__version__,
                "hidden_state_layer": HIDDEN_STATE_LAYER,
                "module_layer_index": MODULE_LAYER_INDEX,
                "format": FORMAT,
                "permutation_order": CANONICAL_PERMUTATION,
                "v_op_dim": int(v_op.shape[0]),
                "v_op_norm": v_op_norm,
                "v_op_source": str(OPERATION_VECTOR_PATH.relative_to(REPO)),
                "v_op_rescale_factor": ov["h1_evaluation"]["primary_rescale_factor"],
                "v_op_construction": ov["h1_evaluation"]["primary_construction"],
                "n_random_directions": N_RANDOM_DIRECTIONS,
                "random_seeds": RANDOM_SEEDS,
                "sample_seed": SAMPLE_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_bootstrap": N_BOOTSTRAP,
                "batch_size": BATCH_SIZE,
                "alpha_min": ALPHA_MIN,
                "alpha_max": ALPHA_MAX,
                "alpha_step": alpha_step,
                "alpha_step_decision": step_reason,
                "items_per_second_baseline_pilot": items_per_sec,
                "smoke": smoke,
            },
            "eval_set": eval_diag,
            "baseline": baseline_metrics,
            "direction_curve": [],
            "random_curve": [],
        }

    state = build_curve_state()
    save_json(state, partial_path)

    # per-item ambig-only correctness, kept in memory (not necessarily every
    # config in the partial file) for the C-H2c bootstrap and for the final
    # artifact's reproducibility section.
    ambig_mask = conditions == "ambig"
    direction_correct_ambig: dict[float, np.ndarray] = {0.0: (baseline_result["chosen"] == correct_positions)[ambig_mask]}
    random_correct_ambig: dict[float, dict[int, np.ndarray]] = {}

    print("[5/7] Direction arm (alpha * v_op)...")
    for alpha in nonzero_alphas:
        scorer.set_delta(alpha * v_op)
        t0 = time.time()
        result = score_eval_set(
            scorer, prompts, correct_positions, BATCH_SIZE,
            baseline_full_log_probs=baseline_result["full_log_probs"], keep_full_log_probs=False,
        )
        elapsed = time.time() - t0
        m = reduce_metrics(result, correct_positions, conditions)
        m["alpha"] = alpha
        m["elapsed_seconds"] = elapsed
        state["direction_curve"].append(m)
        direction_correct_ambig[alpha] = (result["chosen"] == correct_positions)[ambig_mask]
        if verbose:
            print(
                f"      alpha={alpha:+.2f}  acc_ambig={m['accuracy_ambig']:.4f}  "
                f"acc_disambig={m['accuracy_disambig']:.4f}  KL={m['mean_kl_vs_baseline']:.4f}  "
                f"H={m['mean_entropy']:.4f}  ({elapsed:.1f}s)"
            )
        save_json(state, partial_path)

    print("[6/7] Random-direction arm (5 seeds x nonzero alphas)...")
    for alpha in nonzero_alphas:
        per_seed_metrics = {}
        per_seed_correct = {}
        for seed in RANDOM_SEEDS:
            scorer.set_delta(alpha * rng_dirs[seed])
            result = score_eval_set(
                scorer, prompts, correct_positions, BATCH_SIZE,
                baseline_full_log_probs=baseline_result["full_log_probs"], keep_full_log_probs=False,
            )
            m = reduce_metrics(result, correct_positions, conditions)
            per_seed_metrics[seed] = m
            per_seed_correct[seed] = (result["chosen"] == correct_positions)[ambig_mask]

        agg = {}
        for key in ("accuracy_ambig", "accuracy_disambig", "mean_kl_vs_baseline", "mean_entropy", "mean_prob_correct_ambig", "mean_prob_correct_disambig"):
            vals = np.array([per_seed_metrics[s][key] for s in RANDOM_SEEDS], dtype=np.float64)
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std(ddof=1))

        state["random_curve"].append(
            {"alpha": alpha, "aggregate": agg, "per_seed": {str(s): per_seed_metrics[s] for s in RANDOM_SEEDS}}
        )
        random_correct_ambig[alpha] = per_seed_correct
        if verbose:
            print(
                f"      alpha={alpha:+.2f}  acc_ambig(mean over {N_RANDOM_DIRECTIONS} seeds)="
                f"{agg['accuracy_ambig_mean']:.4f}+-{agg['accuracy_ambig_std']:.4f}"
            )
        save_json(state, partial_path)

    print("[7/7] Removing hook, verifying weight integrity, computing C-H2a/b/c...")
    scorer.set_delta(None)
    scorer.remove_hook()
    fingerprint_after = scorer.weight_fingerprint()
    weights_unmodified = fingerprint_before == fingerprint_after
    print(f"      weights unmodified: {weights_unmodified}")
    if not weights_unmodified:
        raise RuntimeError(
            "[INTEGRITY] model weight fingerprint changed after the steering sweep -- the "
            "intervention must be a pure forward-pass hook (spec: 'Integridad del modelo'). "
            "Refusing to write a verdict."
        )

    baseline_acc_ambig = baseline_metrics["accuracy_ambig"]
    baseline_acc_disambig = baseline_metrics["accuracy_disambig"]

    deltas_ambig = {a: (d["accuracy_ambig"] - baseline_acc_ambig) for a, d in zip(nonzero_alphas, state["direction_curve"])}
    best_alpha = max(deltas_ambig, key=lambda a: deltas_ambig[a])
    best_delta = deltas_ambig[best_alpha]
    c_h2a_pass = best_delta >= C_H2A_THRESHOLD

    best_dir_metrics = next(d for d in state["direction_curve"] if d["alpha"] == best_alpha)
    disambig_drop = baseline_acc_disambig - best_dir_metrics["accuracy_disambig"]
    c_h2b_pass = disambig_drop <= C_H2B_THRESHOLD

    vop_correct = direction_correct_ambig[best_alpha]
    random_correct_by_seed = random_correct_ambig[best_alpha]
    n_ambig_eval = vop_correct.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.zeros(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_ambig_eval, size=n_ambig_eval)
        vop_acc = vop_correct[idx].mean()
        rand_acc_mean = np.mean([random_correct_by_seed[s][idx].mean() for s in RANDOM_SEEDS])
        diffs[i] = vop_acc - rand_acc_mean
    ci_lower, ci_upper = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    diff_mean = float(diffs.mean())
    c_h2c_pass = ci_lower > 0.0

    random_agg_at_best = next(r["aggregate"] for r in state["random_curve"] if r["alpha"] == best_alpha)

    coherence_notes = []
    if best_dir_metrics["mean_entropy"] < baseline_metrics["mean_entropy"] and best_delta <= 0:
        coherence_notes.append(
            "entropy fell at the best alpha but accuracy_ambig did not improve there -- "
            "possible unintended model alteration (SOA-004 coherence criterion)."
        )
    if best_delta > 0 and best_dir_metrics["mean_prob_correct_ambig"] <= baseline_metrics["mean_prob_correct_ambig"]:
        coherence_notes.append(
            "accuracy_ambig improved but mean probability mass on the correct (unknown) "
            "answer did not -- accuracy and probability mass disagree at the best alpha."
        )
    if not coherence_notes:
        coherence_notes.append("no incoherence detected between accuracy, entropy and probability mass at the best alpha.")

    output = dict(state)
    output["criteria"] = {
        "C_H2a_effect": {
            "description": "exists alpha with accuracy_ambig >= 10pp above alpha=0",
            "threshold_points": C_H2A_THRESHOLD,
            "best_alpha": best_alpha,
            "accuracy_ambig_alpha0": baseline_acc_ambig,
            "accuracy_ambig_best_alpha": best_dir_metrics["accuracy_ambig"],
            "delta_points": best_delta,
            "passes": bool(c_h2a_pass),
        },
        "C_H2b_cost": {
            "description": "at best_alpha, accuracy_disambig does not fall more than 5pp",
            "threshold_points": C_H2B_THRESHOLD,
            "alpha": best_alpha,
            "accuracy_disambig_alpha0": baseline_acc_disambig,
            "accuracy_disambig_best_alpha": best_dir_metrics["accuracy_disambig"],
            "drop_points": disambig_drop,
            "passes": bool(c_h2b_pass),
        },
        "C_H2c_random_control": {
            "description": (
                "at best_alpha, v_op's accuracy_ambig improvement over alpha=0 exceeds the "
                "norm-matched random arm's improvement, with a bootstrap CI of the difference "
                "excluding zero (this is the deciding criterion)"
            ),
            "alpha": best_alpha,
            "n_bootstrap": N_BOOTSTRAP,
            "n_ambig_eval_items": int(n_ambig_eval),
            "v_op_accuracy_ambig": float(vop_correct.mean()),
            "random_accuracy_ambig_mean_over_seeds": random_agg_at_best["accuracy_ambig_mean"],
            "random_accuracy_ambig_std_over_seeds": random_agg_at_best["accuracy_ambig_std"],
            "difference_mean": diff_mean,
            "difference_ci95": [ci_lower, ci_upper],
            "excludes_zero_positive": bool(c_h2c_pass),
            "passes": bool(c_h2c_pass),
        },
    }
    output["triangulation_at_best_alpha"] = {
        "alpha": best_alpha,
        "direction": {
            "mean_kl_vs_baseline": best_dir_metrics["mean_kl_vs_baseline"],
            "mean_entropy": best_dir_metrics["mean_entropy"],
            "mean_prob_correct_ambig": best_dir_metrics["mean_prob_correct_ambig"],
            "mean_prob_correct_disambig": best_dir_metrics["mean_prob_correct_disambig"],
        },
        "baseline": {
            "mean_entropy": baseline_metrics["mean_entropy"],
            "mean_prob_correct_ambig": baseline_metrics["mean_prob_correct_ambig"],
            "mean_prob_correct_disambig": baseline_metrics["mean_prob_correct_disambig"],
        },
        "random_mean_over_seeds": {
            "mean_kl_vs_baseline": random_agg_at_best["mean_kl_vs_baseline_mean"],
            "mean_entropy": random_agg_at_best["mean_entropy_mean"],
        },
        "coherence_notes": coherence_notes,
    }
    output["weight_integrity"] = {
        "fingerprint_before": fingerprint_before,
        "fingerprint_after": fingerprint_after,
        "unmodified": weights_unmodified,
        "method": "SHA-256 of raw float32 bytes of embed_tokens/intervened-layer q_proj+down_proj/final norm/lm_head",
    }
    output["reproducibility"] = {
        "per_item_eval_set": [
            {"category": e["category"], "condition": e["condition"], "item_id": e["item_id"], "item_key": e["item_key"]}
            for e in eval_items
        ],
        "per_item_ambig_correct_alpha0": direction_correct_ambig[0.0].astype(int).tolist(),
        "per_item_ambig_correct_v_op_best_alpha": vop_correct.astype(int).tolist(),
        "per_item_ambig_correct_random_best_alpha_by_seed": {
            str(s): random_correct_by_seed[s].astype(int).tolist() for s in RANDOM_SEEDS
        },
    }
    elapsed_total = time.time() - start
    output["runtime_seconds"] = elapsed_total
    output["note"] = (
        "H2 verdicts (C-H2a/b/c) reported without interpretation as success or failure of the "
        "hypothesis (Addendum 8: 'si ninguno se cumple, eso es el resultado'). C-H2c is the "
        "deciding criterion; C-H2a/b passing without C-H2c would only show that perturbing the "
        "residual stream makes the model more evasive, not that v_op specifically encodes the "
        "operation."
    )

    save_json(output, result_path)
    if partial_path.exists() and not smoke:
        partial_path.unlink()
    print(f"\n[SAVE] {result_path}")
    print(f"[DONE] total runtime: {elapsed_total/60:.1f} min")
    print(
        f"\nC-H2a (effect)     : {'PASS' if c_h2a_pass else 'FAIL'}  "
        f"(best alpha={best_alpha:+.2f}, delta={best_delta*100:+.1f}pp)"
    )
    print(
        f"C-H2b (cost)       : {'PASS' if c_h2b_pass else 'FAIL'}  "
        f"(disambig drop={disambig_drop*100:+.1f}pp)"
    )
    print(
        f"C-H2c (random ctrl): {'PASS' if c_h2c_pass else 'FAIL'}  "
        f"(v_op - random = {diff_mean*100:+.1f}pp, CI95=[{ci_lower*100:+.1f}, {ci_upper*100:+.1f}]pp)"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-002 H2 causal steering sweep")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny run (2 items/category/condition, 3 alphas) to validate mechanics; writes to _smoke/, "
        "never touches the canonical steering_h2.json.",
    )
    args = parser.parse_args()
    sys.exit(main(verbose=args.verbose, smoke=args.smoke))
