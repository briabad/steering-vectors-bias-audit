"""TF-IDF lexical baseline for C2 / spec "Evaluación relativa a baselines
declarados". Pure text-in, sklearn-out: no I/O, no GPU, testable directly.
"""
from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(train_texts)
    x_holdout = vectorizer.transform(holdout_texts)

    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(x_train, train_labels)
    probs = classifier.predict_proba(x_holdout)[:, 1]
    return float(roc_auc_score(holdout_labels, probs))
