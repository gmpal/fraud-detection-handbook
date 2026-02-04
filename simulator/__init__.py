"""
Fraud Detection Simulator with Causal Mechanisms

This package provides a modular simulator for generating synthetic credit card
transaction data with explicit causal structures for fraud detection research.

Modules:
    core: Customer/terminal profiles and transaction generation
    causal: Structural causal models and interventions
    fraud_scenarios: Original fraud scenarios (legacy)
    causal_scenarios: Causal fraud scenarios with SCMs
    drift: Concept drift and meta-causal switching
    config: Configuration management
    utils: Helper functions
"""

__version__ = "0.1.0"

from .core import (
    generate_customer_profiles_table,
    generate_terminal_profiles_table,
    generate_dataset,
)

__all__ = [
    "generate_customer_profiles_table",
    "generate_terminal_profiles_table",
    "generate_dataset",
]
