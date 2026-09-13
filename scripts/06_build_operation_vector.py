#!/usr/bin/env python3
"""Build and evaluate the operation-level direction v_op (H1 of EXP-002).

Corresponds to openspec change: construccion-vector-operacion

Reuses the already-measured raw corpus (`existence_gate_raw.jsonl`, 287,976
observations) instead of re-running the ~2h GPU evaluation (design.md D1):
- reconstructs item identity positionally,
- pairs P+/P- 1:1 by (category, template, question_polarity) [design.md D2,
  corrected 2026-09-06 after the leak abort -- see `EXP-002/hypothesis.md`
  Addendum 6; spec: "Contraste emparejado 1:1 con el enunciado controlado"],
- extracts activations for the resulting items in ONE pass,
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
from sklearn.metrics import roc_auc_score

from bbq_gate.application.contrast_builder import (
    attach_polarity_and_text,
    build_real_item_index,
    load_raw_rows,
    resolve_contrast_dataset,
    resolve_pairs,
    resolve_transfer_dataset,
)
from bbq_gate.application.vector_construction import (
    BASELINE_LEAK_THRESHOLD,
    best_baseline_margin_check,
    check_baselines_for_leakage,
    item_key,
    layer_auc_curve,
    MarginCheck,
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
    bootstrap_margin_ci,
    cohens_d,
    contrastive_direction,
    cosine,
    pca_direction,
    permutation_test,
    project,
    random_direction_cosine_floor,
    reading_vector,
    rescale_by_activation_norm,
    split_half_cosine_ceiling,
    vectors_are_constant,
)
from bbq_gate.domain.lexical_baseline import lexical_baseline_scores
from bbq_gate.domain.raw_reconstruction import reconstruct_items
from bbq_gate.infrastructure.activation_store import load_activations, save_activations
from bbq_gate.infrastructure.activations import QwenActivationExtractor

FORMAT = "chat"  # primary format; reproduces 1063/22910 (P+/P-) and 564 pairs (corrected key)
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

CONTROL_FORMAT = "plano"  # pairing-only replica, no activation extraction (Addendum 5)

OUTPUT_DIR = REPO / "data" / "experiments" / "EXP-002_bbq_stereotype_direction" / "output"
RAW_PATH = OUTPUT_DIR / "existence_gate_raw.jsonl"
ACTIVATIONS_PATH = OUTPUT_DIR / "operation_vector_activations.npz"
ACTIVATIONS_LAYER0_MEANPOOL_PATH = OUTPUT_DIR / "operation_vector_layer0_meanpool.npz"
RESULT_PATH = OUTPUT_DIR / "operation_vector.json"


def check_disk_space(min_gb: int = 5) -> bool:
    stat = shutil.disk_usage(REPO)
    free_gb = stat.free / (1024**3)
    print(f"[DISK] {free_gb:.1f} GB free (need {min_gb} GB)")
    return free_gb >= min_gb


def build_contrast(reconstructed: list, real_index: dict, format_name: str, verbose: bool) -> dict:
    """Sections 1: pairing, residue, reserved split, for a given prompt format.

    design.md D2 (corrected 2026-09-06, Addendum 6): positives/negatives are
    enriched with `question_polarity`/`question_text` from the real BBQ
    dataset (`attach_polarity_and_text`) BEFORE pairing, because the
    corrected key is `(categoría, question_index, question_polarity)`, not
    `(categoría, question_index)` alone.
    """
    positives, negatives = derive_operation_sets(reconstructed, format_name)
    print(f"      [{format_name}] P+ = {len(positives)}   P- = {len(negatives)}")

    positives = attach_polarity_and_text(positives, real_index)
    negatives = attach_polarity_and_text(negatives, real_index)

    pairs, residue, stats = match_pairs_by_template(positives, negatives)
    zero_neg_templates = templates_without_negatives(stats)
    print(f"      [{format_name}] emparejados 1:1 (clave corregida) = {len(pairs)}   residuo = {len(residue)}")
    if verbose:
        print(f"      [{format_name}] grupos sin negativos: {zero_neg_templates}")

    contrast_ds, transfer_ds = split_reserved(pairs, residue)
    print(
        f"      [{format_name}] ajuste (no reservado) = {len(contrast_ds.pairs)}   "
        f"transferencia (reservado, INTACTO) = {len(transfer_ds.pairs)}"
    )

    residue_adjustable = [r for r in contrast_ds.residue]
    negatives_adjustable = [n for n in negatives if n.category not in RESERVED_CATEGORIES]
    used_negatives = [neg for _pos, neg in pairs]
    residue_pairs = match_residue_by_category(residue_adjustable, negatives_adjustable, used_negatives)
    print(f"      [{format_name}] v_B (residuo emparejado por categoría) = {len(residue_pairs)}")

    n_fused = len(contrast_ds.pairs) + len(residue_pairs)
    print(
        f"      [{format_name}] conjunto de trabajo FUSIONADO (Addendum 5/7, protocolo del "
        f"residuo -> 'fusionar') = {len(contrast_ds.pairs)} (clave estricta, no reservado) + "
        f"{len(residue_pairs)} (residuo por categoría) = {n_fused}"
    )

    return {
        "contrast_ds": contrast_ds,
        "transfer_ds": transfer_ds,
        "residue_pairs": residue_pairs,
        "pairing_stats": {
            "format": format_name,
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "n_paired": len(pairs),
            "n_residue": len(residue),
            "n_non_reserved_paired": len(contrast_ds.pairs),
            "n_residue_matched_by_category": len(residue_pairs),
            "n_fused_working_set": n_fused,
            "groups_without_negatives": [[str(x) for x in t] for t in zero_neg_templates],
            "category_composition_positive_side": category_composition([p for p, _ in pairs]),
            "category_composition_negative_side": category_composition([n for _, n in pairs]),
        },
    }


def extract_all_activations(
    contrast_examples, residue_examples, verbose: bool, reuse_cache: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Section 3: one extraction pass over every distinct item needed.

    Produces TWO activation dicts from the SAME GPU pass (one model load):
    - the main array, all 29 layers at the LAST token (used for direction
      construction, layer curve, residue protocol -- unchanged behaviour);
    - the layer-0 MEAN-POOLED-over-all-tokens array, used ONLY for the
      embeddings baseline (design.md D2 Addendum 6: the last-token position
      is a constant turn marker under 'chat', so it cannot serve as that
      baseline -- see `QwenActivationExtractor.extract_layer0_mean_pooled`).

    If `reuse_cache` and both previously-saved files already cover every
    needed key, skip the GPU pass entirely and load them from disk. Used
    only for iterating on the downstream (non-GPU) analysis without repeating
    the extraction each time; the real, from-scratch run does not need this
    (task 6.2 measures the real, uncached runtime separately via the
    extraction step's own timer, persisted in metadata either way).
    """
    print("[2/6] Loading real BBQ text for the resolved items...")
    all_examples = list(contrast_examples) + list(residue_examples)

    unique_items: dict[str, object] = {}
    for ex in all_examples:
        unique_items[item_key(ex.positive)] = ex.positive
        unique_items[item_key(ex.negative)] = ex.negative
    print(f"      {len(unique_items)} distinct items to extract")

    if reuse_cache and ACTIVATIONS_PATH.exists() and ACTIVATIONS_LAYER0_MEANPOOL_PATH.exists():
        cached_keys, cached_array, cached_metadata = load_activations(ACTIVATIONS_PATH)
        cached_keys_l0, cached_array_l0, _cached_metadata_l0 = load_activations(
            ACTIVATIONS_LAYER0_MEANPOOL_PATH
        )
        if set(cached_keys) >= set(unique_items.keys()) and set(cached_keys_l0) >= set(
            unique_items.keys()
        ):
            print(f"[3/6] Reusing cached activations from {ACTIVATIONS_PATH} and {ACTIVATIONS_LAYER0_MEANPOOL_PATH}")
            # Persisted on disk in fp16 (spec/task 3.3); upcast to float32 for
            # in-memory arithmetic -- see note below on why fp16 reductions
            # (mean/SVD) are unsafe.
            cached_array_f32 = cached_array.astype(np.float32)
            activations_by_key = {k: cached_array_f32[i] for i, k in enumerate(cached_keys)}
            cached_array_l0_f32 = cached_array_l0.astype(np.float32)
            layer0_meanpool_by_key = {k: cached_array_l0_f32[i, 0] for i, k in enumerate(cached_keys_l0)}
            return activations_by_key, layer0_meanpool_by_key, cached_metadata
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
    l0_meanpool_vectors = []
    start = time.time()
    for i, key in enumerate(keys):
        item = unique_items[key]
        prompt, _ = build_prompt(item, FORMAT, CANONICAL_PERMUTATION, tokenizer)
        layer_vectors = extractor.extract(prompt)
        if n_layers is None:
            n_layers = len(layer_vectors)
        vectors.append(np.stack(layer_vectors))
        l0_meanpool_vectors.append(extractor.extract_layer0_mean_pooled(prompt))
        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(keys) - i - 1) / rate
            print(f"      {i + 1}/{len(keys)} extracted ({rate:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"      extraction done: {len(keys)} items, {n_layers} layers, {elapsed:.1f}s")

    activations_array_f16 = np.stack(vectors).astype(np.float16)
    l0_meanpool_array_f16 = np.stack(l0_meanpool_vectors).astype(np.float16)[:, None, :]  # (n, 1, dim)

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

    l0_metadata = dict(metadata)
    l0_metadata["position_policy"] = "mean_pool_all_tokens"
    l0_metadata["n_layers"] = 1
    save_activations(ACTIVATIONS_LAYER0_MEANPOOL_PATH, keys, l0_meanpool_array_f16, l0_metadata)
    print(
        f"      saved {ACTIVATIONS_LAYER0_MEANPOOL_PATH} "
        f"({ACTIVATIONS_LAYER0_MEANPOOL_PATH.stat().st_size / 1e6:.1f} MB)"
    )

    # Persisted on disk in fp16 (spec/task 3.3: "~310 MB"). In-memory
    # arithmetic (mean/diff/SVD) uses float32: float16 reductions over ~175-
    # 350 items overflow silently for the later layers' large-magnitude
    # "outlier" dimensions (observed: layer 27 max |value| ~334, well within
    # a single fp16 value's range but NOT within range once ~300 such values
    # are SUMMED during np.mean's float16 accumulation), producing Inf/NaN
    # downstream. `numpy.linalg.svd` also flatly rejects float16 input.
    activations_array_f32 = activations_array_f16.astype(np.float32)
    activations_by_key = {key: activations_array_f32[i] for i, key in enumerate(keys)}
    l0_meanpool_array_f32 = l0_meanpool_array_f16.astype(np.float32)
    layer0_meanpool_by_key = {key: l0_meanpool_array_f32[i, 0] for i, key in enumerate(keys)}

    return activations_by_key, layer0_meanpool_by_key, metadata


def evaluate_h1(activations_by_key, layer0_meanpool_by_key, contrast_examples, n_layers, verbose: bool) -> dict:
    """Sections 4: split, layer curves, freeze, holdout evaluation, baselines,
    permutation test, effect size, bootstrap.

    `contrast_examples` here is the FUSED working set (strict-key pairs +
    category-matched residue, per Addendum 5/7's 'fusionar' verdict) when
    called from `main`, not the strict-key pairs alone.

    Task 4.6 (revised, Addendum 7 Rev.1-2): aborts -- raising before
    returning, so no artifact is written -- only if the bootstrap CI of the
    margin AUC(direction) - AUC(best baseline) over the holdout includes
    zero. The old absolute-threshold check is retained purely as a
    diagnostic print, not as the gate.
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
    #
    # Embeddings baseline (design.md D2 Addendum 6, spec "Evaluación relativa
    # a baselines declarados"): MEAN-POOLED over all prompt tokens, not the
    # last token. Under 'chat' + add_generation_prompt=True the last token is
    # a fixed turn marker identical across every item, making layer 0 there
    # a constant vector and its AUC exactly 0.5 by construction -- not by
    # absence of signal. `layer0_meanpool_by_key` comes from a SEPARATE
    # embedding-table lookup (no transformer block involved at all).
    layer0_pairs_adjuste = [
        (layer0_meanpool_by_key[item_key(ex.positive)], layer0_meanpool_by_key[item_key(ex.negative)])
        for ex in adjuste_combined
    ]
    layer0_direction = contrastive_direction(layer0_pairs_adjuste)
    layer0_holdout_pos = [layer0_meanpool_by_key[item_key(ex.positive)] for ex in split.holdout]
    layer0_holdout_neg = [layer0_meanpool_by_key[item_key(ex.negative)] for ex in split.holdout]

    layer0_is_degenerate = vectors_are_constant(layer0_holdout_pos + layer0_holdout_neg)
    if layer0_is_degenerate:
        # Spec scenario "Baseline de embeddings no degenerado": signal it as
        # an INVALID baseline instead of reporting a spurious AUC 0.5 as "no
        # discriminative capacity" -- those are different claims.
        layer0_auc = float("nan")
        print("      [WARN] baseline capa-0 (mean-pooled) es CONSTANTE en el holdout -> baseline inválido")
    else:
        layer0_auc = auc_of_direction(layer0_direction, layer0_holdout_pos, layer0_holdout_neg)

    adjuste_texts_pos = [ex.positive.context + " " + ex.positive.question for ex in adjuste_combined]
    adjuste_texts_neg = [ex.negative.context + " " + ex.negative.question for ex in adjuste_combined]
    adjuste_texts = adjuste_texts_pos + adjuste_texts_neg
    adjuste_labels = [1] * len(adjuste_combined) + [0] * len(adjuste_combined)
    holdout_texts_pos = [ex.positive.context + " " + ex.positive.question for ex in split.holdout]
    holdout_texts_neg = [ex.negative.context + " " + ex.negative.question for ex in split.holdout]
    holdout_texts = holdout_texts_pos + holdout_texts_neg
    holdout_labels = [1] * len(split.holdout) + [0] * len(split.holdout)
    # Fit ONCE on the ajuste set; reuse the same fitted model's per-item
    # scores both for the reported AUC and for the margin bootstrap below
    # (Addendum 7: the bootstrap resamples holdout PAIRS, it does not refit
    # the classifier per replicate).
    lexical_scores_holdout = lexical_baseline_scores(adjuste_texts, adjuste_labels, holdout_texts, seed=SEED)
    lexical_pos_scores = lexical_scores_holdout[: len(split.holdout)]
    lexical_neg_scores = lexical_scores_holdout[len(split.holdout) :]
    lexical_auc = float(roc_auc_score(holdout_labels, lexical_scores_holdout))

    baselines = {"lexical_tfidf": lexical_auc, "layer0_embeddings": layer0_auc}
    print(f"      baseline léxico AUC = {lexical_auc:.4f}   baseline capa-0 AUC = {layer0_auc:.4f}")

    # Retired absolute threshold (Addendum 7, Revision 1): kept ONLY as a
    # diagnostic print, never as the pipeline's gate. It could not tell a
    # real lexical leak (baseline ~= direction) apart from the intrinsic
    # lexical signal expected for this contrast (the model's behaviour
    # genuinely depends on who is named in the item).
    threshold_flags = check_baselines_for_leakage(baselines)
    if threshold_flags:
        flagged = ", ".join(f"{c.name}={c.auc:.4f}" for c in threshold_flags)
        print(
            f"      [INFO] (no gate, retired Addendum 7 Rev.1) baseline(s) above the old "
            f"absolute threshold {BASELINE_LEAK_THRESHOLD}: {flagged} -- decision uses the margin gate below"
        )
    else:
        print(f"      [INFO] no baseline exceeds the retired absolute threshold {BASELINE_LEAK_THRESHOLD} (not used as gate)")

    # Margin gate (Addendum 7, Revisions 1-2): margin = AUC(direction) -
    # AUC(baseline), with a 95% bootstrap CI resampling holdout PAIRS
    # jointly for direction and baseline scores. Abort only if the margin
    # over the hardest-to-beat ("mejor") baseline is not distinguishable
    # from zero.
    dir_pos_scores = [project(v, primary_direction) for v in holdout_pos]
    dir_neg_scores = [project(v, primary_direction) for v in holdout_neg]

    margin_checks: list[MarginCheck] = []
    lex_lo, lex_hi, lex_mean = bootstrap_margin_ci(
        dir_pos_scores, dir_neg_scores, lexical_pos_scores, lexical_neg_scores,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    margin_checks.append(
        MarginCheck(
            baseline_name="lexical_tfidf", baseline_auc=lexical_auc,
            margin_mean=lex_mean, margin_ci_lower=lex_lo, margin_ci_upper=lex_hi,
        )
    )
    print(f"      margen dirección - léxico = {lex_mean:+.4f}  IC95%=[{lex_lo:+.4f}, {lex_hi:+.4f}]")

    if not layer0_is_degenerate:
        layer0_pos_scores = [project(v, layer0_direction) for v in layer0_holdout_pos]
        layer0_neg_scores = [project(v, layer0_direction) for v in layer0_holdout_neg]
        l0_lo, l0_hi, l0_mean = bootstrap_margin_ci(
            dir_pos_scores, dir_neg_scores, layer0_pos_scores, layer0_neg_scores,
            n_bootstrap=N_BOOTSTRAP, seed=SEED,
        )
        margin_checks.append(
            MarginCheck(
                baseline_name="layer0_embeddings", baseline_auc=layer0_auc,
                margin_mean=l0_mean, margin_ci_lower=l0_lo, margin_ci_upper=l0_hi,
            )
        )
        print(f"      margen dirección - capa0  = {l0_mean:+.4f}  IC95%=[{l0_lo:+.4f}, {l0_hi:+.4f}]")
    else:
        print("      [INFO] baseline capa-0 inválido (constante en el holdout): excluido del gate de margen")

    best_check = best_baseline_margin_check(margin_checks)
    print(
        f"      mejor baseline (más difícil de batir) = {best_check.baseline_name} "
        f"(AUC={best_check.baseline_auc:.4f})  margen IC95%=[{best_check.margin_ci_lower:+.4f}, "
        f"{best_check.margin_ci_upper:+.4f}]"
    )
    if not best_check.excludes_zero:
        raise RuntimeError(
            f"[TASK 4.6, revised Addendum 7 Rev.1-2] Margin of the direction over the best "
            f"baseline ({best_check.baseline_name}, AUC={best_check.baseline_auc:.4f}) is NOT "
            f"distinguishable from zero: 95% bootstrap CI=[{best_check.margin_ci_lower:+.4f}, "
            f"{best_check.margin_ci_upper:+.4f}] includes 0. Stopping per the revised guardrail "
            f"instead of writing a verdict this sample cannot support -- no artifact is written."
        )

    print("[6/6] Permutation test, Cohen's d, bootstrap CI...")
    diffs = [pos - neg for pos, neg in adjuste_pairs_primary]
    perm_result = permutation_test(diffs, holdout_pos, holdout_neg, n_permutations=N_PERMUTATIONS, seed=SEED)
    d = cohens_d(holdout_pos, holdout_neg, primary_direction)
    ci_lower, ci_upper = bootstrap_auc_ci(
        holdout_pairs_primary, primary_direction, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )
    print(f"      p={perm_result.p_value:.4f}  d={d:.4f}  AUC 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    # The three constructions of SOA-004, each at ITS OWN frozen layer
    # (`chosen_layers`), rebuilt on the same combined ajuste set and
    # evaluated on the same untouched holdout -- so the artifact carries
    # all three direction vectors, not only the primary one (leader
    # instruction, this round: "las tres direcciones").
    directions_all = {
        PRIMARY_CONSTRUCTION: {
            "layer": primary_layer,
            "vector": primary_direction.tolist(),
            "rescale_factor": rescaled.rescale_factor,
            "holdout_auc": direction_auc,
        }
    }
    for name, construct_fn in CONSTRUCTIONS.items():
        if name == PRIMARY_CONSTRUCTION:
            continue
        layer = chosen_layers[name]
        adj_pairs_this = pairs_at_layer(adjuste_combined, activations_by_key, layer)
        raw_direction_this = construct_fn(adj_pairs_this)
        pool_this = [v for pair in adj_pairs_this for v in pair]
        rescaled_this = rescale_by_activation_norm(raw_direction_this, pool_this)
        hold_pairs_this = pairs_at_layer(split.holdout, activations_by_key, layer)
        hold_pos_this = [p for p, _ in hold_pairs_this]
        hold_neg_this = [n for _, n in hold_pairs_this]
        auc_this = auc_of_direction(rescaled_this.vector, hold_pos_this, hold_neg_this)
        directions_all[name] = {
            "layer": layer,
            "vector": rescaled_this.vector.tolist(),
            "rescale_factor": rescaled_this.rescale_factor,
            "holdout_auc": auc_this,
        }
        print(f"      [{name}] layer={layer}  holdout AUC={auc_this:.4f} (non-primary, reported for completeness)")

    return {
        "split_sizes": {
            "sub_adjuste": len(split.sub_adjuste),
            "layer_selection": len(split.layer_selection),
            "holdout": len(split.holdout),
        },
        "layer_curves": curves,
        "chosen_layers": chosen_layers,
        "directions": directions_all,
        "primary_construction": PRIMARY_CONSTRUCTION,
        "primary_layer": primary_layer,
        "primary_rescale_factor": rescaled.rescale_factor,
        "holdout_auc": direction_auc,
        "baselines": baselines,
        "layer0_baseline_valid": not layer0_is_degenerate,
        "layer0_baseline_pooling": "mean_pool_all_tokens",
        "layer0_baseline_interpretation": (
            "AUC of the embeddings layer (layer 0), mean-pooled over all prompt tokens, on "
            "the SAME holdout as the direction. This layer has not passed through any "
            "transformer block: it is the separability available BEFORE the model computes "
            "anything (Addendum 7, 'la mayor parte de la separabilidad es léxica')."
        ),
        "baseline_margins": {
            c.baseline_name: {
                "baseline_auc": c.baseline_auc,
                "margin_mean": c.margin_mean,
                "margin_ci95": [c.margin_ci_lower, c.margin_ci_upper],
                "excludes_zero": c.excludes_zero,
            }
            for c in margin_checks
        },
        "margin_gate": {
            "best_baseline": best_check.baseline_name,
            "decision": "continue (margin CI over best baseline excludes zero)",
            "note": (
                "Replaces the retired absolute threshold of 0.65 (Addendum 7, Revision 1): "
                "aborts (and writes no artifact) only if the margin's bootstrap CI over the "
                "hardest-to-beat baseline includes 0. C2 is reported, not adjudicated "
                "pass/fail on a fixed 0.10 margin (Addendum 7, Revision 2)."
            ),
        },
        "retired_absolute_threshold_diagnostic": {
            "threshold": BASELINE_LEAK_THRESHOLD,
            "flagged_baselines": [{"name": c.name, "auc": c.auc} for c in threshold_flags],
            "note": "Informational only; does not gate the pipeline (Addendum 7, Revision 1).",
        },
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

    print("[1/6] Reconstructing item identity from raw JSONL...")
    rows = load_raw_rows(RAW_PATH)
    print(f"      {len(rows):,} raw observations read")
    reconstructed = list(reconstruct_items(rows).values())
    print(f"      {len(reconstructed):,} items reconstructed")

    print("[1/6] Loading real BBQ text + polarity for all 11 categories...")
    real_index = build_real_item_index(verbose=verbose)

    print(f"[1/6] Pairing (primary format = {FORMAT!r})...")
    contrast_data = build_contrast(reconstructed, real_index, FORMAT, verbose)
    contrast_ds = contrast_data["contrast_ds"]
    transfer_ds = contrast_data["transfer_ds"]
    residue_pairs = contrast_data["residue_pairs"]

    print(f"[1/6] Pairing control replica (format = {CONTROL_FORMAT!r}, Addendum 5, no extraction)...")
    control_contrast_data = build_contrast(reconstructed, real_index, CONTROL_FORMAT, verbose)

    contrast_examples = resolve_contrast_dataset(contrast_ds, real_index)  # v_A candidates (strict key)
    residue_examples = resolve_pairs(tuple(residue_pairs), real_index)  # v_B candidates (residue, by category)
    transfer_examples = resolve_transfer_dataset(transfer_ds, real_index)  # not used for fitting

    # ADDENDUM 5/7: the residue protocol's pre-registered verdict (measured
    # in a prior run: cos(v_A, v_B) = 0.9199 against ceiling 0.9667 / floor
    # -0.0009, threshold 80% of ceiling = 0.7734) is "fusionar". The working
    # set for H1 evaluation (split, layer selection, holdout, direction
    # construction) is therefore the UNION of the strict-key pairs and the
    # category-matched residue -- NOT the strict-key pairs alone. Reserved
    # categories are excluded from BOTH components already (design.md D6:
    # `contrast_ds.pairs` and `residue_pairs` are both non-reserved by
    # construction), so the union stays isolated from H3 transfer material.
    fused_examples = list(contrast_examples) + list(residue_examples)
    print(
        f"[FUSION] Addendum 5/7 protocolo del residuo -> 'fusionar': conjunto de trabajo = "
        f"{len(contrast_examples)} (v_A, clave estricta) + {len(residue_examples)} "
        f"(v_B, residuo por categoría) = {len(fused_examples)} pares no reservados"
    )

    activations_by_key, layer0_meanpool_by_key, extraction_metadata = extract_all_activations(
        contrast_examples, residue_examples, verbose, reuse_cache=reuse_cache
    )
    n_layers = extraction_metadata["n_layers"]

    h1_result = evaluate_h1(
        activations_by_key, layer0_meanpool_by_key, fused_examples, n_layers, verbose
    )
    primary_direction = h1_result.pop("_primary_direction")
    primary_layer = h1_result.pop("_primary_layer")

    # Residue protocol reported for the record at the layer frozen by the
    # FUSED evaluation above: v_A/v_B stay the ORIGINAL, disjoint, pre-fusion
    # sets (spec "Conjuntos disjuntos") -- fusing them would make v_A and v_B
    # overlap with the fitting set and defeat the comparison's purpose.
    residue_result = residue_protocol(activations_by_key, contrast_examples, residue_examples, primary_layer)

    elapsed = time.time() - start
    output = {
        "metadata": extraction_metadata,
        "pairing": contrast_data["pairing_stats"],
        "pairing_control_replica": control_contrast_data["pairing_stats"],
        "fused_working_set": {
            "n_strict_key_pairs": len(contrast_examples),
            "n_residue_pairs": len(residue_examples),
            "n_fused": len(fused_examples),
            "note": (
                "Addendum 5's pre-registered residue protocol (measured: "
                "cos(v_A, v_B) = 0.9199, ceiling 0.9667 +/-0.0155, floor "
                "-0.0009 +/-0.0167, threshold 80% of ceiling = 0.7734) "
                "recommended 'fusionar'. This is the set used for the "
                "split, layer selection, holdout, direction construction "
                "and evaluation in `h1_evaluation` below -- not the "
                "strict-key pairs alone."
            ),
        },
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
        "limitations": [
            (
                "Selección hacia plantillas con abstenciones disponibles: el emparejamiento "
                "1:1 (D2) descarta preferentemente las plantillas de sesgo más marcado, que "
                "casi nunca dejan negativos ('unknown') disponibles para emparejar (p. ej. "
                "poligamia, idolatría, alcohol -- ver hypothesis.md Addendum 5). El conjunto "
                "de ajuste, incluso fusionado con el residuo, sigue sobrerrepresentando "
                "relativamente los casos donde el modelo SÍ se abstiene con más frecuencia "
                "frente a los estereotipos más extremos, donde casi nunca lo hace."
            ),
            (
                "Posible contaminación de BBQ en el entrenamiento: BBQ es público desde 2022 "
                "y puede formar parte de los datos de entrenamiento de Qwen 2.5. Si el modelo "
                "memorizó el corpus, una respuesta 'unknown' podría deberse a memorización en "
                "vez de razonamiento sobre la ambigüedad, lo que no es descartable con el "
                "diseño de esta PoC (hypothesis.md, Addendum 1)."
            ),
        ],
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
