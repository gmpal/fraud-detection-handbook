"""
Causal inference components for fraud detection

Modules:
    scm: Structural Causal Models
    graphs: Causal graph definitions
    mechanisms: Causal mechanisms for fraud scenarios
    interventions: Do-calculus and counterfactual queries
"""

from .scm import StructuralCausalModel

__all__ = ["StructuralCausalModel"]
