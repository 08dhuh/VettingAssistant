from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ViolationType(Enum):
    RULE_OF_N = "rule_of_n"
    RULE_OF_10 = "rule_of_10"
    DOMINANCE_D50 = "dominance_d50"
    DOMINANCE_D67 = "dominance_d67"
    # Regression-specific violations
    INSUFFICIENT_DF = "insufficient_degrees_of_freedom"
    NUMERIC_VARIABLE = "numeric_variable_disclosure_risk"
    HIGH_R_SQUARED = "high_r_squared_with_intercept"
    OBSERVATION_DIFFERENCE = "observation_difference"


class WarningType(Enum):
    PERCENTAGE_MISMATCH = "percentage_mismatch"
    METADATA_INCOMPLETE = "metadata_incomplete"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    # Regression-specific warnings
    LOW_R_SQUARED = "low_r_squared"
    LOW_DF_WITH_CONTINUOUS = "low_df_with_continuous"
    VARIABLE_TYPE_UNKNOWN = "variable_type_unknown"
    VARIABLE_MISMATCH = "variable_mismatch"


@dataclass
class Violation:
    violation_type: ViolationType
    row: int
    column: str
    value: float
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.violation_type.value} at row {self.row}, column '{self.column}': {self.value}"


@dataclass
class Warning:
    warning_type: WarningType
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.warning_type.value}: {self.message}"


@dataclass
class ValidationResult:
    passed: bool
    violations: List[Violation]
    warnings: List[Warning] = field(default_factory=list)
    total_checks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        status = 'PASS' if self.passed else 'FAIL'
        return f"{status}: {len(self.violations)} violations, {len(self.warnings)} warnings"

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class DatasetMetadata:
    """Metadata extracted from description document"""
    population: Optional[str] = None
    method_of_analysis: Optional[str] = None
    datasets_used: Optional[List[str]] = field(default_factory=list)
    data_description: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)  # variable_name: description
    percentage_columns: List[str] = field(default_factory=list)  # Columns identified as percentages
    sample_size_column: Optional[str] = None  # Column containing sample sizes/counts
    raw_description: str = ""
    # Regression-specific fields
    regression_type: Optional[str] = None  # "OLS", "logit", "probit", etc.
    variable_types: Dict[str, str] = field(default_factory=dict)  # variable_name: "continuous"/"binary"/"unknown"
    outcome_variable: Optional[str] = None  # What the regression is predicting

    def is_complete(self) -> bool:
        """Check if all required metadata is present"""
        return all([
            self.population,
            self.method_of_analysis,
            self.datasets_used,
            self.data_description,
            self.variables
        ])


@dataclass
class RegressionCoefficient:
    """A single coefficient from a regression model output"""
    variable_name: str
    coefficient: float
    std_error: Optional[float] = None
    significance_level: int = 0  # 0=none, 1=*, 2=**, 3=***
    variable_type: Optional[str] = None  # "continuous", "binary", "categorical", or None

    def __str__(self) -> str:
        stars = "*" * self.significance_level
        return f"{self.variable_name}: {self.coefficient}{stars}"


@dataclass
class RegressionMetadata:
    """Metadata for regression model outputs"""
    n_observations: int
    r_squared: float
    coefficients: List[RegressionCoefficient] = field(default_factory=list)
    has_intercept: bool = False
    regression_type: str = "OLS"

    _INTERCEPT_NAMES = {"constant", "intercept", "_cons"}

    def __post_init__(self):
        """Detect intercept from coefficient names if not explicitly set"""
        if not self.has_intercept and self.coefficients:
            self.has_intercept = any(
                coef.variable_name.lower() in self._INTERCEPT_NAMES
                for coef in self.coefficients
            )

    @property
    def n_independent_vars(self) -> int:
        """Count of coefficients excluding intercept"""
        if not self.has_intercept:
            return len(self.coefficients)
        return sum(
            1 for coef in self.coefficients
            if coef.variable_name.lower() not in self._INTERCEPT_NAMES
        )

    @property
    def degrees_of_freedom(self) -> int:
        """Degrees of freedom: n - k - 1"""
        return self.n_observations - self.n_independent_vars - 1

    @property
    def has_continuous_variable(self) -> bool:
        """True if any coefficient has variable_type='continuous'"""
        return any(
            coef.variable_type == "continuous"
            for coef in self.coefficients
        )


@dataclass
class RegressionValidationResult:
    """Validation result specific to regression outputs"""
    passed: bool
    violations: List[Violation]
    warnings: List[Warning] = field(default_factory=list)
    regression_metadata: Optional[RegressionMetadata] = None
    description_metadata: Optional[DatasetMetadata] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_id: Optional[str] = None  # Identifier for multi-model validation

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status}: {len(self.violations)} violations, {len(self.warnings)} warnings"

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class ModelComparisonResult:
    """Result of comparing observation counts between two regression models"""
    model_a_id: str  # e.g., "Model 1" or filename
    model_b_id: str
    n_a: Optional[int]
    n_b: Optional[int]
    difference: Optional[int]
    is_subset: Optional[bool]
    passed: Optional[bool]  # None if cannot be determined (e.g., suppressed counts)
    message: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.passed is False else "CHECK")
        return f"{self.model_a_id} vs {self.model_b_id}: {status} - {self.message}"


@dataclass
class MultiModelValidationResult:
    """Result of validating multiple regression models"""
    individual_results: List[RegressionValidationResult]
    comparison_results: List[ModelComparisonResult]
    overall_passed: bool

    @property
    def summary(self) -> str:
        individual_passed = sum(1 for r in self.individual_results if r.passed)
        comparison_failed = sum(1 for r in self.comparison_results if r.passed is False)
        status = "PASS" if self.overall_passed else "FAIL"
        return (
            f"{status}: {individual_passed}/{len(self.individual_results)} models passed, "
            f"{comparison_failed} comparison violation(s)"
        )

    @property
    def has_comparison_violations(self) -> bool:
        return any(r.passed is False for r in self.comparison_results)