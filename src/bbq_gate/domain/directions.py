"""Direction algebra: construction, rescaling, projection, and significance.

Pure numpy/sklearn, no I/O, no GPU (design.md D5, Goals: "el álgebra de
direcciones... testeable sin GPU"). Implements the three constructions of
SOA-004 (reading vector, contrastive, PCA), norm rescaling, cosine/projection
primitives, the permutation test WITH reajuste (spec: "Significancia por
permutación con reajuste"), Cohen's d, and item-level bootstrap.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

Vector = np.ndarray
Pair = tuple[Vector, Vector]  # (positive_activation, negative_activation)


# --------------------------------------------------------------------------
# Construction (task 2.1, spec "Tres construcciones de la dirección...")
# --------------------------------------------------------------------------


def reading_vector(vectors: Sequence[Vector]) -> Vector:
    """SOA-004 eq. 2: mean of a SINGLE set of activations, no contrast at all.

    This is deliberately the "peor" (worst) construction per SOA-004's own
    table: it never looks at the negative class, so any topical composition
    difference between what happens to be in `vectors` and everything else
    leaks directly into the result.
    """
    if not vectors:
        raise ValueError("reading_vector requires at least one vector")
    return np.stack(vectors).astype(np.float64).mean(axis=0)


def contrastive_direction(pairs: Sequence[Pair]) -> Vector:
    """SOA-004 eq. 3: mean of PAIRED differences H(P+) - H(P-).

    Spec "Diferencias emparejadas": the input MUST be an explicit sequence of
    (positive, negative) pairs, not two independently-sized sets -- see the
    module-level test `test_pairs_confound_reading_vector_but_not_contrastive`
    for why an unpaired difference of set means (as `reading_vector` applied
    to two skewed conjuntos sueltos) can smuggle in a topical confound that
    the paired construction, by design, cannot.

    Inputs are cast to float64 regardless of their original dtype: summing a
    few hundred fp16 activations (the on-disk persistence dtype, task 3.3)
    can silently overflow float16's ~65504 range for later-layer "outlier"
    dimensions, producing Inf/NaN with no exception raised until much later
    (observed in the real pipeline: layer 27 activations with |value| ~334,
    summed over ~350 items during `np.mean` in fp16 precision).
    """
    if not pairs:
        raise ValueError("contrastive_direction requires at least one pair")
    diffs = np.stack([np.asarray(pos, dtype=np.float64) - np.asarray(neg, dtype=np.float64) for pos, neg in pairs])
    return diffs.mean(axis=0)


def pca_direction(pairs: Sequence[Pair]) -> Vector:
    """SOA-004 eq. 4: first principal component of the paired differences.

    Sign is aligned with the mean difference direction so it points from
    negative to positive (PCA's sign is otherwise arbitrary).

    Inputs are cast to float64: `numpy.linalg.svd` outright rejects float16
    (`TypeError: array type float16 is unsupported in linalg`), which is the
    dtype activations are persisted in (task 3.3).
    """
    if not pairs:
        raise ValueError("pca_direction requires at least one pair")
    diffs = np.stack([np.asarray(pos, dtype=np.float64) - np.asarray(neg, dtype=np.float64) for pos, neg in pairs])
    mean_diff = diffs.mean(axis=0)
    centered = diffs - mean_diff
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    if np.dot(pc1, mean_diff) < 0:
        pc1 = -pc1
    return pc1


# --------------------------------------------------------------------------
# Rescaling (task 2.2, spec "Comparabilidad de escala")
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RescaledDirection:
    """A direction paired with the rescale factor that produced it, so the
    factor is always available to persist (spec: "el factor de reescalado
    aplicado queda registrado")."""

    vector: Vector
    rescale_factor: float


def rescale_by_activation_norm(
    direction: Vector, activations: Sequence[Vector]
) -> RescaledDirection:
    """Rescale `direction` to the mean norm of real activations at this layer.

    PCA directions come out unit-norm from SVD ("la trampa de escala",
    SOA-004 §A.1); reading_vector/contrastive_direction are already
    differences/means of real activations, so their natural norm is already
    at the real scale, but they are rescaled too (factor ~1.0) purely so all
    three methods' factors are recorded uniformly (task 2.2: "verificar que
    las tres direcciones quedan en el mismo orden de magnitud y que el
    factor aplicado se registra").

    Args:
        direction: The raw constructed direction.
        activations: The pool of real activations at this layer, used to
            compute the target norm (their mean L2 norm).

    Returns:
        RescaledDirection with the rescaled vector and the applied factor.

    Raises:
        ValueError: if `direction` has zero norm (cannot rescale) or
            `activations` is empty.
    """
    if not activations:
        raise ValueError("rescale_by_activation_norm requires at least one activation")
    current_norm = float(np.linalg.norm(np.asarray(direction, dtype=np.float64)))
    if current_norm == 0.0:
        raise ValueError("direction has zero norm, cannot rescale")
    # Cast each activation to float64 before computing its norm: squaring a
    # SINGLE large-magnitude float16 value (e.g. ~334, seen at layer 27 of
    # the real Qwen 2.5 7B activations) already overflows fp16's ~65504
    # range elementwise, silently producing `inf` (see directions.py module
    # notes on `contrastive_direction`/`pca_direction` for the same class of
    # bug found in the real pipeline).
    target_norm = float(np.mean([np.linalg.norm(np.asarray(a, dtype=np.float64)) for a in activations]))
    factor = target_norm / current_norm
    return RescaledDirection(vector=direction * factor, rescale_factor=factor)


# --------------------------------------------------------------------------
# Projection, normalization, cosine (task 2.3)
# --------------------------------------------------------------------------


def normalize(vector: Vector) -> Vector:
    """Unit-normalize a vector. Raises on zero-norm input.

    Casts to float64 internally: a raw fp16 activation vector (the on-disk
    persistence dtype, task 3.3) can overflow `np.linalg.norm` to `inf` on a
    SINGLE large-magnitude element (observed: layer 27 values ~334, whose
    square already exceeds fp16's ~65504 range) well before any reduction
    across items happens.
    """
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def project(vector: Vector, direction: Vector) -> float:
    """Scalar projection of `vector` onto (not-necessarily-unit) `direction`.

    A zero-norm `direction` returns 0.0 for every input rather than raising.
    This is a real, expected case here: a degenerate construction (e.g. a
    contrastive direction built from pairs whose difference is EXACTLY zero
    at some layer -- observed at layer 0/embeddings under the 'chat' format,
    whose fixed `add_generation_prompt` suffix makes the last-token embedding
    identical for every item regardless of content) has no discriminative
    direction to project onto. Returning a constant 0.0 for every item makes
    `auc_of_direction` correctly report AUC 0.5 (no separation) instead of
    crashing the whole layer curve -- which is the semantically right
    baseline outcome (spec: "la capa de embeddings no ha atravesado ningún
    bloque del modelo").

    `direction` is cast to float64 internally for the same overflow reason
    as `normalize`; `vector` (often a raw fp16 activation) is left as-is --
    `np.dot` against a float64 `direction` promotes the whole computation to
    float64 before multiplying, so it does not need a separate cast here.
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return 0.0
    return float(np.dot(vector, direction) / norm)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors.

    Casts both inputs to float64 internally (see `normalize`'s docstring for
    why: fp16 activations can overflow `np.linalg.norm` on a single
    large-magnitude element).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vectors_are_constant(vectors: Sequence[Vector], atol: float = 1e-6) -> bool:
    """True if every vector in `vectors` is (numerically) identical.

    Spec "Baseline de embeddings no degenerado": "si resultara constante, el
    sistema lo señala como baseline inválido en vez de reportarlo como
    capacidad discriminativa nula". Used to catch the exact failure mode
    diagnosed in `EXP-002/hypothesis.md` Addendum 6 -- under
    `add_generation_prompt=True`, the LAST token of a 'chat'-format prompt is
    a turn marker identical across every item, making layer 0 at that
    position a constant vector and its AUC exactly 0.5 by construction, not
    by absence of signal. Cast to float64 for the same overflow reason as
    `normalize`/`cosine` (fp16 activations, task 3.3).
    """
    if len(vectors) < 2:
        return False
    arr = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    return bool(np.allclose(arr, arr[0], atol=atol))


# --------------------------------------------------------------------------
# AUC helper shared by permutation test, layer selection, and baselines
# --------------------------------------------------------------------------


def auc_of_direction(direction: Vector, positives: Sequence[Vector], negatives: Sequence[Vector]) -> float:
    """ROC-AUC of separating `positives` from `negatives` by their scalar
    projection onto `direction`."""
    scores = np.concatenate(
        [np.array([project(p, direction) for p in positives]), np.array([project(n, direction) for n in negatives])]
    )
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------------
# Permutation test WITH reajuste (task 2.4, spec "Significancia por
# permutación con reajuste")
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    """Result of a permutation significance test.

    Attributes:
        observed_auc: AUC of the real (unpermuted) direction on the fixed holdout.
        null_aucs: The null distribution of holdout AUCs from reconstructed
            directions under label permutation.
        p_value: (# null >= observed + 1) / (n_permutations + 1).
    """

    observed_auc: float
    null_aucs: tuple[float, ...]
    p_value: float


def permutation_test(
    diffs: Sequence[Vector],
    holdout_positives: Sequence[Vector],
    holdout_negatives: Sequence[Vector],
    n_permutations: int = 1000,
    seed: int | None = None,
) -> PermutationResult:
    """Permutation test for the contrastive direction, WITH reajuste.

    Spec "Nulo correcto": for each of `n_permutations` repetitions, the
    direction is RECONSTRUCTED from the permuted assignment -- here,
    implemented as a random sign-flip of each per-pair difference vector
    (equivalent to relabeling which side of each pair is "positive"), which
    respects the paired structure of the ajuste set. Each reconstructed
    direction is evaluated, unchanged, on the FIXED real holdout, building a
    null distribution of achievable holdout AUCs under "no real pairing
    signal, just this construction process".

    This is the null of interest per design.md's risk "El vector puede
    codificar tema pese al emparejamiento": if a random (label-flipped)
    reconstruction of the SAME pairs also tends to separate the holdout well
    (e.g. because of a category/template confound shared between ajuste and
    holdout), the null distribution widens accordingly and the observed AUC
    correctly stops looking exceptional. Omitting reajuste (fixing the
    direction and only permuting evaluation labels, see
    `permutation_test_without_reajuste`) cannot detect this and is the
    prohibited shortcut the spec names explicitly.

    Args:
        diffs: Per-pair difference vectors (positive - negative) from the
            ajuste set used to build the real direction.
        holdout_positives: Positive-class activations from the (never
            permuted) holdout.
        holdout_negatives: Negative-class activations from the holdout.
        n_permutations: Number of sign-flip permutations.
        seed: RNG seed for reproducibility.

    Returns:
        PermutationResult with the observed AUC, the null distribution, and
        the one-sided p-value.
    """
    if not diffs:
        raise ValueError("permutation_test requires at least one diff")
    diffs_arr = np.stack(diffs).astype(np.float64)  # see contrastive_direction on fp16 overflow
    observed_direction = diffs_arr.mean(axis=0)
    observed_auc = auc_of_direction(observed_direction, holdout_positives, holdout_negatives)

    rng = random.Random(seed)
    null_aucs = []
    for _ in range(n_permutations):
        signs = np.array([rng.choice((-1.0, 1.0)) for _ in range(len(diffs_arr))])
        permuted_direction = (diffs_arr * signs[:, None]).mean(axis=0)
        null_aucs.append(auc_of_direction(permuted_direction, holdout_positives, holdout_negatives))

    p_value = (sum(1 for v in null_aucs if v >= observed_auc) + 1) / (n_permutations + 1)
    return PermutationResult(observed_auc=observed_auc, null_aucs=tuple(null_aucs), p_value=p_value)


def permutation_test_without_reajuste(
    diffs: Sequence[Vector],
    holdout_positives: Sequence[Vector],
    holdout_negatives: Sequence[Vector],
    n_permutations: int = 1000,
    seed: int | None = None,
) -> PermutationResult:
    """THE PROHIBITED SHORTCUT (spec: "Permutar sin reconstruir la dirección
    ... devuelve significancia siempre"). Exists ONLY for the regression test
    required by task 2.4 (`tests/test_domain_directions.py::
    test_permutation_without_reajuste_gives_lower_p_under_confound`). Never
    call this from the real pipeline (`application`/`scripts`).

    Fits the direction ONCE from the true (unpermuted) diffs and never
    reconstructs it. "Permutation" here only reshuffles the HOLDOUT labels
    and re-scores the SAME fixed direction's fixed holdout projections. This
    tests a narrower, different null ("is this exact fixed vector better
    than chance at this exact holdout") that cannot detect a construction
    process vulnerable to a shared ajuste/holdout confound: its null
    distribution is always the label-permutation-of-fixed-scores baseline
    (~0.5, narrow variance), regardless of whether the construction process
    itself is confound-prone. When ajuste and holdout share a confound (e.g.
    category/template composition), the correct method's null widens to
    reflect that risk while this one does not, so this shortcut yields a
    systematically smaller (falsely more significant) p-value.
    """
    if not diffs:
        raise ValueError("permutation_test_without_reajuste requires at least one diff")
    diffs_arr = np.stack(diffs).astype(np.float64)  # see contrastive_direction on fp16 overflow
    fixed_direction = diffs_arr.mean(axis=0)
    observed_auc = auc_of_direction(fixed_direction, holdout_positives, holdout_negatives)

    scores = np.concatenate(
        [
            np.array([project(p, fixed_direction) for p in holdout_positives]),
            np.array([project(n, fixed_direction) for n in holdout_negatives]),
        ]
    )
    true_labels = np.concatenate([np.ones(len(holdout_positives)), np.zeros(len(holdout_negatives))])

    rng = np.random.default_rng(seed)
    null_aucs = []
    for _ in range(n_permutations):
        shuffled_labels = rng.permutation(true_labels)
        null_aucs.append(float(roc_auc_score(shuffled_labels, scores)))

    p_value = (sum(1 for v in null_aucs if v >= observed_auc) + 1) / (n_permutations + 1)
    return PermutationResult(observed_auc=observed_auc, null_aucs=tuple(null_aucs), p_value=p_value)


# --------------------------------------------------------------------------
# Effect size and item-level bootstrap (task 2.5)
# --------------------------------------------------------------------------


def cohens_d(positives: Sequence[Vector], negatives: Sequence[Vector], direction: Vector) -> float:
    """Cohen's d between the two classes' scalar projections onto `direction`."""
    pos_scores = np.array([project(p, direction) for p in positives])
    neg_scores = np.array([project(n, direction) for n in negatives])
    n1, n2 = len(pos_scores), len(neg_scores)
    if n1 < 2 or n2 < 2:
        raise ValueError("cohens_d requires at least 2 items per group")
    var1, var2 = pos_scores.var(ddof=1), neg_scores.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0.0:
        raise ValueError("pooled standard deviation is zero, Cohen's d undefined")
    return float((pos_scores.mean() - neg_scores.mean()) / pooled_std)


def bootstrap_auc_ci(
    pairs: Sequence[Pair],
    direction: Vector,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for the AUC of a FIXED `direction`, resampling holdout
    ITEMS (pairs), not observations (task 2.5: "verificar que duplicar
    artificialmente los datos no estrecha el intervalo").

    Args:
        pairs: Holdout (positive, negative) pairs.
        direction: The already-selected, fixed direction to evaluate.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: RNG seed.

    Returns:
        (lower, upper) bounds of the AUC confidence interval.
    """
    if not pairs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(pairs) for _ in range(len(pairs))]
        positives = [p for p, _ in sample]
        negatives = [n for _, n in sample]
        try:
            values.append(auc_of_direction(direction, positives, negatives))
        except ValueError:
            continue  # a degenerate resample (all-same-class scores) is skipped
    if not values:
        return float("nan"), float("nan")
    values.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * len(values))
    upper_idx = min(int((1 - alpha / 2) * len(values)), len(values) - 1)
    return values[lower_idx], values[upper_idx]


# --------------------------------------------------------------------------
# Margin bootstrap (hypothesis.md Addendum 7, Revisions 1-2): replaces the
# retired absolute BASELINE_LEAK_THRESHOLD gate with a check on whether the
# margin of the direction over a baseline is distinguishable from zero.
# --------------------------------------------------------------------------


def bootstrap_margin_ci(
    direction_pos_scores: Sequence[float],
    direction_neg_scores: Sequence[float],
    baseline_pos_scores: Sequence[float],
    baseline_neg_scores: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for margin = AUC(direction) - AUC(baseline) on the same
    fixed holdout, resampling holdout PAIRS jointly for both AUCs.

    Addendum 7, Revision 1: the pipeline used to abort on an absolute
    baseline AUC threshold (0.65), which cannot tell a real lexical leak
    (baseline ~= direction) apart from an expected, intrinsic lexical signal
    (this contrast's behaviour genuinely depends on who is named). The
    replacement asks the question that actually matters: is the direction's
    advantage over the baseline distinguishable from sampling noise? Each
    bootstrap replicate resamples a single set of pair indices ONCE and uses
    it to recompute BOTH AUCs, so the margin reflects the SAME resampled
    holdout rather than the difference of two independently-resampled AUCs
    (which would overstate the interval's width).

    Args:
        direction_pos_scores / direction_neg_scores: per-pair scalar
            projections of the fixed, already-selected direction on the
            holdout positives / negatives (pair i in each array corresponds
            to the same holdout pair).
        baseline_pos_scores / baseline_neg_scores: per-pair scalar scores of
            the fixed, already-fitted baseline model on the SAME holdout
            pairs, same order.
        n_bootstrap: number of resamples.
        confidence: confidence level for the two-sided interval.
        seed: RNG seed.

    Returns:
        (ci_lower, ci_upper, mean) of the margin distribution. All NaN if no
        replicate produced a valid AUC (e.g. an empty holdout).
    """
    n = len(direction_pos_scores)
    if not (n == len(direction_neg_scores) == len(baseline_pos_scores) == len(baseline_neg_scores)):
        raise ValueError(
            "direction_pos_scores, direction_neg_scores, baseline_pos_scores and "
            "baseline_neg_scores must all have the same length (one holdout pair each)"
        )
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    dps = np.asarray(direction_pos_scores, dtype=np.float64)
    dns = np.asarray(direction_neg_scores, dtype=np.float64)
    bps = np.asarray(baseline_pos_scores, dtype=np.float64)
    bns = np.asarray(baseline_neg_scores, dtype=np.float64)
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    rng = random.Random(seed)
    margins = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        dir_scores = np.concatenate([dps[idx], dns[idx]])
        base_scores = np.concatenate([bps[idx], bns[idx]])
        try:
            auc_dir = roc_auc_score(labels, dir_scores)
            auc_base = roc_auc_score(labels, base_scores)
        except ValueError:
            continue  # degenerate resample (e.g. every score tied); skip
        margins.append(auc_dir - auc_base)

    if not margins:
        return float("nan"), float("nan"), float("nan")
    margins.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * len(margins))
    upper_idx = min(int((1 - alpha / 2) * len(margins)), len(margins) - 1)
    return margins[lower_idx], margins[upper_idx], float(np.mean(margins))


# --------------------------------------------------------------------------
# Residue protocol anchors (task 5.2-5.3, design.md D3)
# --------------------------------------------------------------------------


def random_direction_cosine_floor(
    dim: int, n_samples: int = 1000, seed: int | None = None
) -> tuple[float, float]:
    """Floor anchor: mean and std of cosine similarity between pairs of
    independent random directions in R^dim. Expected ~0, std ~1/sqrt(dim)
    (spec: "el valor esperado entre direcciones aleatorias")."""
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(n_samples):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        cosines.append(cosine(a, b))
    arr = np.array(cosines)
    return float(arr.mean()), float(arr.std())


def split_half_cosine_ceiling(
    pairs: Sequence[Pair],
    construct_fn: "callable[[Sequence[Pair]], Vector]",
    n_splits: int = 50,
    seed: int | None = None,
) -> tuple[float, float]:
    """Ceiling anchor: cosine between directions built from two disjoint
    random halves of `pairs`, averaged over `n_splits` partitions (spec: "el
    obtenido entre dos mitades del conjunto emparejado").

    Args:
        pairs: The full set of paired items to split.
        construct_fn: A direction-construction function taking a sequence of
            pairs (e.g. `contrastive_direction`).
        n_splits: Number of random disjoint 50/50 partitions to average over.
        seed: RNG seed.

    Returns:
        (mean, std) of the split-half cosine over the partitions.
    """
    if len(pairs) < 4:
        raise ValueError("split_half_cosine_ceiling requires at least 4 pairs")
    rng = random.Random(seed)
    cosines = []
    indices = list(range(len(pairs)))
    for _ in range(n_splits):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        half_a = [pairs[i] for i in shuffled[:half]]
        half_b = [pairs[i] for i in shuffled[half:]]
        direction_a = construct_fn(half_a)
        direction_b = construct_fn(half_b)
        cosines.append(cosine(direction_a, direction_b))
    arr = np.array(cosines)
    return float(arr.mean()), float(arr.std())
