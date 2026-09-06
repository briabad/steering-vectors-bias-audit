#!/usr/bin/env python3
"""Build and evaluate the operation-level direction v_op (H1 of EXP-002).

Corresponds to openspec change: construccion-vector-operacion

Reuses the already-measured raw corpus (`existence_gate_raw.jsonl`, 287,976
observations) instead of re-running the ~2h GPU evaluation (design.md D1):
- reconstructs item identity positionally,
- pairs P+/P- 1:1 by (category, template) [spec: "Contraste emparejado 1:1"],
- extracts activations for the ~1500 resulting items in ONE pass,
- selects a layer without touching the holdout,
- evaluates H1 (AUC vs baselines, permutation test with reajuste, effect
  size, bootstrap CI),
- runs the residue protocol (v_A vs v_B, design.md D3).

Reserved categories (Race_x_gender, Gender_identity, Sexual_orientation) are
isolated by type (design.md D6) and never touched by any fitting decision
here; they are declared, not used.

Usage:
    python scripts/06_build_operation_vector.py [-v]

Runtime note: ~1500-2000 forward passes of short prompts on a single RTX
4090 -- expected minutes, not hours (proposal.md "Coste").
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from bbq_gate.application.contrast_builder import (
    build_real_item_index,
    load_raw_rows,
    resolve_contrast_dataset,
    resolve_pairs,
    resolve_transfer_dataset,
)
from bbq_gate.application.vector_construction import (
    BASELINE_LEAK_THRESHOLD,
    check_baselines_for_leakage,
    item_key,
    layer_auc_curve,
    pairs_at_layer,
    select_best_layer,
    stratified_split,
)
from bbq_gate.domain.contrast import (
    RESERVED_CATEGORIES,
    category_composition,
    derive_operation_sets,
    match_pairs_by_template,
    match_residue_by_category,
    split_reserved,
    templates_without_negatives,
)
from bbq_gate.domain.directions import (
    auc_of_direction,
    bootstrap_auc_ci,
    cohens_d,
    contrastive_direction,
    cosine,
    pca_direction,
    permutation_test,
    random_direction_cosine_floor,
    reading_vector,
    rescale_by_activation_norm,
    split_half_cosine_ceiling,
)
from bbq_gate.domain.lexical_baseline import lexical_baseline_auc
from bbq_gate.domain.raw_reconstruction import reconstruct_items
from bbq_gate.infrastructure.activation_store import load_activations, save_activations
from bbq_gate.infrastructure.activations import QwenActivationExtractor

FORMAT = "chat"  # primary format; reproduces the 1063/22910/752/311 figures
CANONICAL_PERMUTATION = [0, 1, 2]
SEED = 20260906
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
CONSTRUCTIONS = {
    "reading_vector": lambda pairs: reading_vector([p for p, _ in pairs]),
    "contrastive": contrastive_direction,
    "pca": pca_direction,
}
PRIMARY_CONSTRUCTION = "contrastive"

OUTPUT_DIR = REPO / "data" / "experiments" / "EXP-002_bbq_stereotype_direction" / "output"
RAW_PATH = OUTPUT_DIR / "existence_gate_raw.jsonl"
ACTIVATIONS_PATH = OUTPUT_DIR / "operation_vector_activations.npz"
RESULT_PATH = OUTPUT_DIR / "operation_vector.json"


def check_disk_space(min_gb: int = 5) -> bool:
    stat = shutil.disk_usage(REPO)
    free_gb = stat.free / (1024**3)
    print(f"[DISK] {free_gb:.1f} GB free (need {min_gb} GB)")
    return free_gb >= min_gb


def build_contrast(verbose: bool) -> dict:
    """Sections 1: crude reconstruction, pairing, residue, reserved split."""
    print("[1/6] Reconstructing item identity from raw JSONL...")
    rows = load_raw_rows(RAW_PATH)
    print(f"      {len(rows):,} raw observations read")
    reconstructed = list(reconstruct_items(rows).values())
    print(f"      {len(reconstructed):,} items reconstructed")

    positives, negatives = derive_operation_sets(reconstructed, FORMAT)
    print(f"      P+ = {len(positives)}   P- = {len(negatives)}")

    pairs, residue, stats = match_pairs_by_template(positives, negatives)
    zero_neg_templates = templates_without_negatives(stats)
    print(f"      emparejados 1:1 = {len(pairs)}   residuo = {len(residue)}")
    print(f"      plantillas sin negativos: {zero_neg_templates}")

    contrast_ds, transfer_ds = split_reserved(pairs, residue)
    print(
        f"      ajuste (no reservado) = {len(contrast_ds.pairs)}   "
        f"transferencia (reservado, INTACTO) = {len(transfer_ds.pairs)}"
    )

    residue_adjustable = [r for r in contrast_ds.residue]
    negatives_adjustable = [n for n in negatives if n.category not in RESERVED_CATEGORIES]
    used_negatives = [neg for _pos, neg in pairs]
    residue_pairs = match_residue_by_category(residue_adjustable, negatives_adjustable, used_negatives)
    print(f"      v_B (residuo emparejado por categoría) = {len(residue_pairs)}")

    return {
        "contrast_ds": contrast_ds,
        "transfer_ds": transfer_ds,
        "residue_pairs": residue_pairs,
        "pairing_stats": {
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "n_paired": len(pairs),
            "n_residue": len(residue),
            "templates_without_negatives": [list(t) for t in zero_neg_templates],
            "category_composition_positive_side": category_composition([p for p, _ in pairs]),
            "category_composition_negative_side": category_composition([n for _, n in pairs]),
        },
    }


def extract_all_activations(
    contrast_examples, residue_examples, verbose: bool, reuse_cache: bool = False
) -> tuple[dict[str, np.ndarray], dict]:
    """Section 3: one extraction pass over every distinct item needed.

    If `reuse_cache` and a previously-saved activations file already covers
    every needed key, skip the GPU pass entirely and load it from disk. Used
    only for iterating on the downstream (non-GPU) analysis without repeating
    a ~1 minute extraction each time; the real, from-scratch run does not
    need this (task 6.2 measures the real, uncached runtime separately via
    the extraction step's own timer, persisted in metadata either way).
    """
    print("[2/6] Loading real BBQ text for the resolved items...")
    all_examples = list(contrast_examples) + list(residue_examples)

    unique_items: dict[str, object] = {}
    for ex in all_examples:
        unique_items[item_key(ex.positive)] = ex.positive
        unique_items[item_key(ex.negative)] = ex.negative
    print(f"      {len(unique_items)} distinct items to extract")

    if reuse_cache and ACTIVATIONS_PATH.exists():
        cached_keys, cached_array, cached_metadata = load_activations(ACTIVATIONS_PATH)
        if set(cached_keys) >= set(unique_items.keys()):
            print(f"[3/6] Reusing cached activations from {ACTIVATIONS_PATH}")
            # Persisted on disk in fp16 (spec/task 3.3); upcast to float32 for
            # in-memory arithmetic -- see note below on why fp16 reductions
            # (mean/SVD) are unsafe.
            cached_array_f32 = cached_array.astype(np.float32)
            activations_by_key = {k: cached_array_f32[i] for i, k in enumerate(cached_keys)}
            return activations_by_key, cached_metadata
        print("      cache found but incomplete for the current item set; re-extracting")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing to run on CPU (DEC-002)")

    print("[3/6] Loading Qwen 2.5 7B Instruct and extracting activations...")
    extractor = QwenActivationExtractor()
    tokenizer = extractor.tokenizer

    from bbq_gate.infrastructure.prompts import build_prompt
    keys = sorted(unique_items.keys())
    n_layers = None
    vectors = []
    start = time.time()
    for i, key in enumerate(keys):
        item = unique_items[key]
        prompt, _ = build_prompt(item, FORMAT, CANONICAL_PERMUTATION, tokenizer)
        layer_vectors = extractor.extract(prompt)
        if n_layers is None:
            n_layers = len(layer_vectors)
        vectors.append(np.stack(layer_vectors))
        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(keys) - i - 1) / rate
            print(f"      {i + 1}/{len(keys)} extracted ({rate:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"      extraction done: {len(keys)} items, {n_layers} layers, {elapsed:.1f}s")

    activations_array_f16 = np.stack(vectors).astype(np.float16)

    metadata = {
        "model": extractor.model_id,
        "revision": extractor.revision,
        "dtype": "float16",
        "position_policy": "last_token",
        "seed": SEED,
        "format": FORMAT,
        "permutation_order": CANONICAL_PERMUTATION,
        "n_layers": n_layers,
        "extraction_seconds": elapsed,
    }
    save_activations(ACTIVATIONS_PATH, keys, activations_array_f16, metadata)
    print(f"      saved {ACTIVATIONS_PATH} ({ACTIVATIONS_PATH.stat().st_size / 1e6:.1f} MB)")

    # Persisted on disk in fp16 (spec/task 3.3: "~310 MB"). In-memory
    # arithmetic (mean/diff/SVD) uses float32: float16 reductions over ~175-
    # 350 items overflow silently for the later layers' large-magnitude
    # "outlier" dimensions (observed: layer 27 max |value| ~334, well within
    # a single fp16 value's range but NOT within range once ~300 such values
    # are SUMMED during np.mean's float16 accumulation), producing Inf/NaN
    # downstream. `numpy.linalg.svd` also flatly rejects float16 input.
    activations_array_f32 = activations_array_f16.astype(np.float32)
    activations_by_key = {key: activations_array_f32[i] for i, key in enumerate(keys)}

    return activations_by_key, metadata


def evaluate_h1(activations_by_key, contrast_examples, n_layers, verbose: bool) -> dict:
    """Sections 4: split, layer curves, freeze, holdout evaluation, baselines,
    permutation test, effect size, bootstrap. Task 4.6: abort if a baseline
    leaks.
    """
    print("[4/6] Splitting sub-ajuste / selección-de-capa / holdout...")
    split = stratified_split(contrast_examples, fractions=(0.5, 0.25, 0.25), seed=SEED)
    print(
        f"      sub_adjuste={len(split.sub_adjuste)}  "
        f"layer_selection={len(split.layer_selection)}  holdout={len(split.holdout)}"
    )

    print("[5/6] Layer curves for the three constructions...")
    curves = {}
    chosen_layers = {}
    for name, construct_fn in CONSTRUCTIONS.items():
        curve = layer_auc_curve(split.sub_adjuste, split.layer_selection, activations_by_key, construct_fn, n_layers)
        curves[name] = curve
        chosen_layers[name] = select_best_layer(curve)
        if verbose:
            print(f"      {name}: chosen layer={chosen_layers[name]} (selection AUC={curve[chosen_layers[name]]:.4f})")

    # Final direction for the PRIMARY construction: rebuilt on sub_adjuste +
    # layer_selection COMBINED at the frozen layer, evaluated ONLY NOW on
    # the untouched holdout.
    primary_layer = chosen_layers[PRIMARY_CONSTRUCTION]
    adjuste_combined = list(split.sub_adjuste) + list(split.layer_selection)
    adjuste_pairs_primary = pairs_at_layer(adjuste_combined, activations_by_key, primary_layer)
    holdout_pairs_primary = pairs_at_layer(split.holdout, activations_by_key, primary_layer)
    holdout_pos = [p for p, _ in holdout_pairs_primary]
    holdout_neg = [n for _, n in holdout_pairs_primary]

    primary_direction_raw = CONSTRUCTIONS[PRIMARY_CONSTRUCTION](adjuste_pairs_primary)
    activation_pool = [v for pair in adjuste_pairs_primary for v in pair]
    rescaled = rescale_by_activation_norm(primary_direction_raw, activation_pool)
    primary_direction = rescaled.vector

    direction_auc = auc_of_direction(primary_direction, holdout_pos, holdout_neg)
    print(f"      PRIMARY ({PRIMARY_CONSTRUCTION}) holdout AUC (layer {primary_layer}) = {direction_auc:.4f}")

    # Baselines, on the SAME holdout.
    layer0_pairs_adjuste = pairs_at_layer(adjuste_combined, activations_by_key, 0)
    layer0_direction = contrastive_direction(layer0_pairs_adjuste)
    layer0_pairs_holdout = pairs_at_layer(split.holdout, activations_by_key, 0)
    layer0_auc = auc_of_direction(
        layer0_direction, [p for p, _ in layer0_pairs_holdout], [n for _, n in layer0_pairs_holdout]
    )

    adjuste_texts = [ex.positive.context + " " + ex.positive.question for ex in adjuste_combined] + [
        ex.negative.context + " " + ex.negative.question for ex in adjuste_combined
    ]
    adjuste_labels = [1] * len(adjuste_combined) + [0] * len(adjuste_combined)
    holdout_texts = [ex.positive.context + " " + ex.positive.question for ex in split.holdout] + [
        ex.negative.context + " " + ex.negative.question for ex in split.holdout
    ]
    holdout_labels = [1] * len(split.holdout) + [0] * len(split.holdout)
    lexical_auc = lexical_baseline_auc(adjuste_texts, adjuste_labels, holdout_texts, holdout_labels, seed=SEED)

    baselines = {"lexical_tfidf": lexical_auc, "layer0_embeddings": layer0_auc}
    print(f"      baseline léxico AUC = {lexical_auc:.4f}   baseline capa-0 AUC = {layer0_auc:.4f}")

    leaks = check_baselines_for_leakage(baselines)
    if leaks:
        leak_report = ", ".join(f"{leak.name}={leak.auc:.4f}" for leak in leaks)
        raise RuntimeError(
            f"[TASK 4.6] POSSIBLE LEAK: baseline(s) exceed AUC {BASELINE_LEAK_THRESHOLD}: {leak_report}. "
            f"Stopping per spec ('Baseline inesperadamente fuerte') instead of continuing."
        )

    print("[6/6] Permutation test, Cohen's d, bootstrap CI...")
    diffs = [pos - neg for pos, neg in adjuste_pairs_primary]
    perm_result = permutation_test(diffs, holdout_pos, holdout_neg, n_permutations=N_PERMUTATIONS, seed=SEED)
    d = cohens_d(holdout_pos, holdout_neg, primary_direction)
    ci_lower, ci_upper = bootstrap_auc_ci(
        holdout_pairs_primary, primary_direction, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )
    print(f"      p={perm_result.p_value:.4f}  d={d:.4f}  AUC 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        "split_sizes": {
            "sub_adjuste": len(split.sub_adjuste),
            "layer_selection": len(split.layer_selection),
            "holdout": len(split.holdout),
        },
        "layer_curves": curves,
        "chosen_layers": chosen_layers,
        "primary_construction": PRIMARY_CONSTRUCTION,
        "primary_layer": primary_layer,
        "primary_rescale_factor": rescaled.rescale_factor,
        "holdout_auc": direction_auc,
        "baselines": baselines,
        "permutation_p_value": perm_result.p_value,
        "permutation_observed_auc": perm_result.observed_auc,
        "permutation_null_mean": float(np.mean(perm_result.null_aucs)),
        "cohens_d": d,
        "auc_bootstrap_ci95": [ci_lower, ci_upper],
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "_primary_direction": primary_direction,  # consumed by residue protocol, stripped before saving
        "_primary_layer": primary_layer,
    }


def residue_protocol(activations_by_key, contrast_examples, residue_examples, primary_layer: int) -> dict:
    """Section 5: v_A vs v_B, anchored against floor (random) and ceiling
    (split-half of v_A)."""
    print("[residue] Building v_A / v_B and anchors...")
    v_a_pairs = pairs_at_layer(contrast_examples, activations_by_key, primary_layer)
    v_b_pairs = pairs_at_layer(residue_examples, activations_by_key, primary_layer)

    v_a = contrastive_direction(v_a_pairs)
    v_b = contrastive_direction(v_b_pairs) if v_b_pairs else None

    dim = v_a.shape[0]
    floor_mean, floor_std = random_direction_cosine_floor(dim=dim, n_samples=1000, seed=SEED)
    ceiling_mean, ceiling_std = split_half_cosine_ceiling(
        v_a_pairs, contrastive_direction, n_splits=50, seed=SEED
    )

    result = {
        "n_v_a_pairs": len(v_a_pairs),
        "n_v_b_pairs": len(v_b_pairs),
        "floor_random_cosine_mean": floor_mean,
        "floor_random_cosine_std": floor_std,
        "ceiling_split_half_cosine_mean": ceiling_mean,
        "ceiling_split_half_cosine_std": ceiling_std,
    }
    if v_b is not None:
        cos_ab = cosine(v_a, v_b)
        threshold = 0.8 * ceiling_mean
        recommendation = "fusionar" if cos_ab >= threshold else "mantener separados (residuo = evaluación)"
        result["cosine_v_a_v_b"] = cos_ab
        result["fuse_threshold_80pct_ceiling"] = threshold
        result["recommendation"] = recommendation
        print(f"      cos(v_A, v_B) = {cos_ab:.4f}  floor~{floor_mean:.4f}  ceiling~{ceiling_mean:.4f}")
        print(f"      recomendación: {recommendation}")
    else:
        result["cosine_v_a_v_b"] = None
        result["recommendation"] = "no hay suficientes pares de residuo para construir v_B"
    return result


def main(verbose: bool = False, reuse_cache: bool = False) -> int:
    start = time.time()
    print("[INIT] construccion-vector-operacion")
    if not check_disk_space():
        return 1
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return 1

    contrast_data = build_contrast(verbose)
    contrast_ds = contrast_data["contrast_ds"]
    transfer_ds = contrast_data["transfer_ds"]
    residue_pairs = contrast_data["residue_pairs"]

    real_index = build_real_item_index(verbose=verbose)
    contrast_examples = resolve_contrast_dataset(contrast_ds, real_index)
    residue_examples = resolve_pairs(tuple(residue_pairs), real_index)
    transfer_examples = resolve_transfer_dataset(transfer_ds, real_index)  # not used for fitting

    activations_by_key, extraction_metadata = extract_all_activations(
        contrast_examples, residue_examples, verbose, reuse_cache=reuse_cache
    )
    n_layers = extraction_metadata["n_layers"]

    h1_result = evaluate_h1(activations_by_key, contrast_examples, n_layers, verbose)
    primary_direction = h1_result.pop("_primary_direction")
    primary_layer = h1_result.pop("_primary_layer")

    residue_result = residue_protocol(activations_by_key, contrast_examples, residue_examples, primary_layer)

    elapsed = time.time() - start
    output = {
        "metadata": extraction_metadata,
        "pairing": contrast_data["pairing_stats"],
        "h1_evaluation": h1_result,
        "residue_protocol": residue_result,
        "reserved_categories": {
            "categories": sorted(RESERVED_CATEGORIES),
            "n_transfer_pairs": len(transfer_ds.pairs),
            "used_in_fitting": False,
            "note": (
                "Declared and counted only. No activation of these pairs was "
                "extracted or used in any construction, layer selection, or "
                "adjustment decision in this change (design.md D6)."
            ),
        },
        "primary_direction": primary_direction.tolist(),
        "runtime_seconds": elapsed,
        "note": (
            "H1 verdict reported without interpretation as success or failure "
            "of the change (proposal.md: 'un AUC bajo es un resultado válido')."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVE] {RESULT_PATH}")
    print(f"[DONE] total runtime: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and evaluate the operation-level direction v_op")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse a previously-saved activations file if it covers all needed items "
        "(skips the GPU pass; for iterating on the analysis, not the canonical run).",
    )
    args = parser.parse_args()
    sys.exit(main(verbose=args.verbose, reuse_cache=args.reuse_cache))
