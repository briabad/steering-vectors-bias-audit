"""TF-IDF lexical baseline for C2 / spec "Evaluación relativa a baselines
declarados". Pure text-in, sklearn-out: no I/O, no GPU, testable directly.
"""
from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def lexical_baseline_scores(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    holdout_texts: Sequence[str],
    seed: int = 0,
) -> list[float]:
    """Fit a TF-IDF + logistic regression classifier on `train_texts` and
    return its predicted P(positive) for each `holdout_texts` item, in the
    same order.

    Split out of `lexical_baseline_auc` (unchanged behaviour there) so
    callers that need the SAME fixed fitted model's per-item scores -- e.g.
    the margin bootstrap of `EXP-002/hypothesis.md` Addendum 7 (Revisions
    1-2), which resamples holdout pairs and re-scores them WITHOUT refitting
    the classifier per replicate -- can reuse it instead of only getting a
    single aggregate AUC.

    Args:
        train_texts: Item texts (e.g. context + question) for the ajuste set.
        train_labels: 1 for positive class, 0 for negative.
        holdout_texts: Item texts for the holdout set.
        seed: Random state for the classifier.

    Returns:
        Per-item P(positive) scores for `holdout_texts`, in order.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(train_texts)
    x_holdout = vectorizer.transform(holdout_texts)

    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(x_train, train_labels)
    return classifier.predict_proba(x_holdout)[:, 1].tolist()


def lexical_baseline_auc(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    holdout_texts: Sequence[str],
    holdout_labels: Sequence[int],
    seed: int = 0,
) -> float:
    """Fit a TF-IDF + logistic regression classifier on `train_texts` and
    return its AUC on `holdout_texts`.

    Args:
        train_texts: Item texts (e.g. context + question) for the ajuste set.
        train_labels: 1 for positive class, 0 for negative.
        holdout_texts: Item texts for the holdout set.
        holdout_labels: 1/0 labels for the holdout.
        seed: Random state for the classifier.

    Returns:
        ROC-AUC on the holdout.
    """
    probs = lexical_baseline_scores(train_texts, train_labels, holdout_texts, seed=seed)
    return float(roc_auc_score(holdout_labels, probs))
