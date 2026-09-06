"""BBQ existence gate: measure stereotypical bias direction via latent vectors.

This module orchestrates the evaluation of Qwen 2.5 7B Instruct on the BBQ
(Bias Benchmark for QA) dataset across 11 categories to detect whether the model
exhibits directional stereotypical bias when responding to ambiguous questions.

See docs/architecture.md §9 for the single-domain model.
"""

__version__ = "0.0.1"
