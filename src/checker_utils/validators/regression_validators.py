"""
Validators specific to regression output analysis.

Contains validators for multi-model comparison including:
- Observation count difference checks
"""

from typing import List, Tuple

from ..models import (
    RegressionMetadata,
    ModelComparisonResult,
)


# Intercept names to exclude from variable comparisons
INTERCEPT_NAMES = {"constant", "intercept", "_cons", "(intercept)"}


def validate_observation_differences(
    models: List[Tuple[str, RegressionMetadata]]
) -> List[ModelComparisonResult]:
    """
    Compare all pairs of models for observation count differences.

    Rule: Differences < 10 between subset models are violations.
    - If models have the same observation count: PASS (no differencing risk)
    - If difference < 10 AND one model's variables are a subset of the other: FAIL
    - If difference < 10 AND variables are NOT subsets: PASS (different populations)
    - If difference >= 10: PASS
    - If counts are suppressed (None): Cannot determine, manual check required

    Args:
        models: List of (model_id, RegressionMetadata) tuples

    Returns:
        List of ModelComparisonResult for all pairwise comparisons
    """
    results = []

    for i, (id_a, model_a) in enumerate(models):
        for id_b, model_b in models[i + 1:]:  # Avoid duplicate pairs
            result = _compare_two_models(id_a, model_a, id_b, model_b)
            results.append(result)

    return results


def _compare_two_models(
    id_a: str,
    model_a: RegressionMetadata,
    id_b: str,
    model_b: RegressionMetadata,
) -> ModelComparisonResult:
    """
    Compare observation counts between two models.

    Args:
        id_a: Identifier for model A
        model_a: RegressionMetadata for model A
        id_b: Identifier for model B
        model_b: RegressionMetadata for model B

    Returns:
        ModelComparisonResult with comparison outcome
    """
    n_a = model_a.n_observations
    n_b = model_b.n_observations

    # Check if counts are suppressed/missing
    if n_a is None or n_b is None:
        return ModelComparisonResult(
            model_a_id=id_a,
            model_b_id=id_b,
            n_a=n_a,
            n_b=n_b,
            difference=None,
            is_subset=None,
            passed=None,
            message="Counts suppressed — cannot verify observation difference",
        )

    diff = abs(n_a - n_b)

    # Same sample size — no differencing risk
    if diff == 0:
        return ModelComparisonResult(
            model_a_id=id_a,
            model_b_id=id_b,
            n_a=n_a,
            n_b=n_b,
            difference=0,
            is_subset=None,  # Not relevant when diff is 0
            passed=True,
            message="Same sample size — no differencing risk",
        )

    # Difference >= 10: always safe
    if diff >= 10:
        return ModelComparisonResult(
            model_a_id=id_a,
            model_b_id=id_b,
            n_a=n_a,
            n_b=n_b,
            difference=diff,
            is_subset=None,  # Not checked when diff >= 10
            passed=True,
            message=f"Observation difference of {diff:,} ≥ 10",
        )

    # Difference < 10: need to check subset relationship
    vars_a = _get_variable_names(model_a)
    vars_b = _get_variable_names(model_b)

    is_subset = vars_a.issubset(vars_b) or vars_b.issubset(vars_a)

    if is_subset:
        # Determine which is the subset
        if vars_a.issubset(vars_b) and not vars_b.issubset(vars_a):
            subset_info = f"{id_a} variables ⊂ {id_b} variables"
        elif vars_b.issubset(vars_a) and not vars_a.issubset(vars_b):
            subset_info = f"{id_b} variables ⊂ {id_a} variables"
        else:
            subset_info = "Same variables in both models"

        return ModelComparisonResult(
            model_a_id=id_a,
            model_b_id=id_b,
            n_a=n_a,
            n_b=n_b,
            difference=diff,
            is_subset=True,
            passed=False,
            message=(
                f"Observation difference of {diff} < 10 between subset models. "
                f"{subset_info}. Suppress counts or justify non-comparable populations."
            ),
        )
    else:
        return ModelComparisonResult(
            model_a_id=id_a,
            model_b_id=id_b,
            n_a=n_a,
            n_b=n_b,
            difference=diff,
            is_subset=False,
            passed=True,
            message=f"Observation difference of {diff} < 10, but different variables — not subsets",
        )


def _get_variable_names(model: RegressionMetadata) -> set:
    """
    Extract variable names from model coefficients, excluding intercept.

    Args:
        model: RegressionMetadata

    Returns:
        Set of variable names (lowercase for case-insensitive comparison)
    """
    return {
        coef.variable_name.lower()
        for coef in model.coefficients
        if coef.variable_name.lower() not in INTERCEPT_NAMES
    }
