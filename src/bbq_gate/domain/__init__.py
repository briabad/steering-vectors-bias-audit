"""Domain layer: entities, value objects, and business logic.

This module contains domain entities and core metrics that are testable
without GPU or network access. It declares the Scorer protocol (inverted
dependency: domain defines the contract, infrastructure implements it).
"""
