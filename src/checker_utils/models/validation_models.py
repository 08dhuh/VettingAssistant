from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ViolationType(Enum):
    RULE_OF_N = "rule_of_n"
    DOMINANCE_D50 = "dominance_d50"
    DOMINANCE_D67 = "dominance_d67"


class WarningType(Enum):
    PERCENTAGE_MISMATCH = "percentage_mismatch"
    METADATA_INCOMPLETE = "metadata_incomplete"
    DATA_TYPE_MISMATCH = "data_type_mismatch"


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

    def is_complete(self) -> bool:
        """Check if all required metadata is present"""
        return all([
            self.population,
            self.method_of_analysis,
            self.datasets_used,
            self.data_description,
            self.variables
        ])