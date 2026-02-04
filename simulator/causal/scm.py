"""
Structural Causal Model (SCM) implementation for fraud detection

A Structural Causal Model represents causal relationships using functional equations:
    X_i = f_i(PA_i, U_i)
where PA_i are parents of X_i in the causal graph, and U_i is exogenous noise.
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class Variable:
    """Represents a variable in the SCM"""
    name: str
    is_latent: bool = False
    parents: List[str] = field(default_factory=list)
    mechanism: Optional[Callable] = None
    noise_dist: Optional[Callable] = None

    def __repr__(self):
        latent_str = "*" if self.is_latent else ""
        return f"{self.name}{latent_str}"


class StructuralCausalModel:
    """
    Structural Causal Model for fraud detection scenarios.

    A SCM consists of:
    - Variables: Observable and latent
    - Structural equations: Functional relationships between variables
    - Noise distributions: Exogenous randomness

    Example:
        >>> scm = StructuralCausalModel()
        >>> scm.add_variable("COMPROMISED", is_latent=True,
        ...                  mechanism=lambda: np.random.binomial(1, 0.01))
        >>> scm.add_variable("TX_AMOUNT",
        ...                  parents=["COMPROMISED"],
        ...                  mechanism=lambda compromised, noise:
        ...                      500 if compromised else 100 + noise)
        >>> samples = scm.sample(n=1000)
    """

    def __init__(self, name: str = "SCM"):
        self.name = name
        self.variables: Dict[str, Variable] = {}
        self.topological_order: List[str] = []
        self._is_sorted = False

    def add_variable(
        self,
        name: str,
        parents: Optional[List[str]] = None,
        mechanism: Optional[Callable] = None,
        noise_dist: Optional[Callable] = None,
        is_latent: bool = False
    ):
        """
        Add a variable to the SCM.

        Args:
            name: Variable name
            parents: List of parent variable names (causes)
            mechanism: Function that computes variable from parents and noise
            noise_dist: Function that generates exogenous noise
            is_latent: Whether variable is latent (unobserved)
        """
        if parents is None:
            parents = []

        self.variables[name] = Variable(
            name=name,
            is_latent=is_latent,
            parents=parents,
            mechanism=mechanism,
            noise_dist=noise_dist
        )
        self._is_sorted = False  # Need to recompute topological order

    def _topological_sort(self):
        """Compute topological ordering of variables (parents before children)"""
        if self._is_sorted:
            return

        visited = set()
        order = []

        def visit(var_name):
            if var_name in visited:
                return
            visited.add(var_name)

            # Visit parents first
            if var_name in self.variables:
                for parent in self.variables[var_name].parents:
                    visit(parent)

            order.append(var_name)

        for var_name in self.variables:
            visit(var_name)

        self.topological_order = order
        self._is_sorted = True

    def sample(
        self,
        n_samples: int = 1000,
        interventions: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate samples from the SCM.

        Args:
            n_samples: Number of samples to generate
            interventions: Dictionary of do-operator interventions {var_name: value}
            random_state: Random seed for reproducibility

        Returns:
            DataFrame with sampled values for all variables
        """
        if random_state is not None:
            np.random.seed(random_state)

        if interventions is None:
            interventions = {}

        # Ensure topological order
        self._topological_sort()

        # Initialize samples dictionary
        samples = {}

        # Generate samples in topological order
        for var_name in self.topological_order:
            var = self.variables[var_name]

            # Check if this variable is intervened upon
            if var_name in interventions:
                # Do-operator: Set value directly
                intervention_value = interventions[var_name]
                if callable(intervention_value):
                    samples[var_name] = intervention_value(n_samples)
                else:
                    samples[var_name] = np.full(n_samples, intervention_value)
            else:
                # Normal causal mechanism
                if var.mechanism is None:
                    raise ValueError(f"No mechanism defined for variable {var_name}")

                # Get parent values
                parent_values = {p: samples[p] for p in var.parents if p in samples}

                # Generate noise
                noise = None
                if var.noise_dist is not None:
                    noise = var.noise_dist(n_samples)

                # Apply mechanism
                # Mechanism receives parent values and noise
                if len(parent_values) == 0 and noise is None:
                    # Exogenous variable with no parents
                    samples[var_name] = var.mechanism(n_samples)
                elif noise is not None:
                    samples[var_name] = var.mechanism(**parent_values, noise=noise)
                else:
                    samples[var_name] = var.mechanism(**parent_values)

        # Convert to DataFrame
        df = pd.DataFrame(samples)

        # Only return observable variables by default
        observable_vars = [name for name, var in self.variables.items() if not var.is_latent]
        return df[observable_vars + [name for name in df.columns if name not in observable_vars]]

    def intervene(self, var_name: str, value: Any) -> 'StructuralCausalModel':
        """
        Create a new SCM with do(var_name = value) intervention.

        This performs Pearl's do-operator by removing incoming edges to var_name
        and setting its value.

        Args:
            var_name: Variable to intervene on
            value: Value to set (can be constant or callable)

        Returns:
            New SCM with intervention applied
        """
        # Create copy of SCM
        new_scm = StructuralCausalModel(name=f"{self.name}_do({var_name}={value})")

        # Copy all variables except the intervened one
        for name, var in self.variables.items():
            if name == var_name:
                # Intervened variable: no parents, constant value
                new_scm.add_variable(
                    name=name,
                    parents=[],
                    mechanism=lambda n, v=value: np.full(n, v) if not callable(v) else v(n),
                    is_latent=var.is_latent
                )
            else:
                # Other variables: copy as is
                new_scm.add_variable(
                    name=name,
                    parents=var.parents.copy(),
                    mechanism=var.mechanism,
                    noise_dist=var.noise_dist,
                    is_latent=var.is_latent
                )

        return new_scm

    def counterfactual(
        self,
        evidence: Dict[str, Any],
        intervention: Dict[str, Any],
        query: str,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Answer counterfactual query: "What would Y be if we set X=x, given evidence E?"

        This uses the three-step process:
        1. Abduction: Infer exogenous variables U from evidence
        2. Action: Apply intervention do(X=x)
        3. Prediction: Compute Y under intervention

        Args:
            evidence: Observed values {var_name: value}
            intervention: Interventional values {var_name: value}
            query: Variable to query
            n_samples: Number of samples for approximation

        Returns:
            Array of counterfactual values for query variable

        Note:
            This is a simplified implementation. Full counterfactual inference
            requires inverting the structural equations, which may not always
            be possible analytically.
        """
        # For now, implement simplified version:
        # Sample from intervened distribution
        intervened_scm = self
        for var_name, value in intervention.items():
            intervened_scm = intervened_scm.intervene(var_name, value)

        samples = intervened_scm.sample(n_samples=n_samples)
        return samples[query].values

    def get_causal_graph(self) -> Dict[str, List[str]]:
        """
        Get causal graph as adjacency list.

        Returns:
            Dictionary mapping each variable to its parents
        """
        return {name: var.parents for name, var in self.variables.items()}

    def __repr__(self):
        self._topological_sort()
        var_str = "\n".join([f"  {var}" for var in self.topological_order])
        return f"StructuralCausalModel('{self.name}'):\n{var_str}"


# Helper functions for common mechanisms

def bernoulli_mechanism(p: float) -> Callable:
    """Create a Bernoulli mechanism"""
    return lambda n: np.random.binomial(1, p, n)


def normal_mechanism(mean: float, std: float) -> Callable:
    """Create a Gaussian mechanism"""
    return lambda n: np.random.normal(mean, std, n)


def conditional_mechanism(condition_fn: Callable, true_fn: Callable, false_fn: Callable) -> Callable:
    """
    Create a conditional mechanism: if condition then true_fn else false_fn

    Example:
        >>> mechanism = conditional_mechanism(
        ...     condition_fn=lambda compromised: compromised == 1,
        ...     true_fn=lambda: np.random.normal(500, 100),  # Fraud
        ...     false_fn=lambda: np.random.normal(100, 50)    # Legitimate
        ... )
    """
    def _mechanism(**parent_values):
        # Evaluate condition
        condition = condition_fn(**parent_values)

        # Apply corresponding function
        if isinstance(condition, np.ndarray):
            # Vectorized version
            result = np.where(condition, true_fn(), false_fn())
        else:
            result = true_fn() if condition else false_fn()

        return result

    return _mechanism
