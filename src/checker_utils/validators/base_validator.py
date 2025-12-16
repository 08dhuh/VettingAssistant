from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from ..models import Violation, Warning, DatasetMetadata


class BaseValidator(ABC):
    """Abstract base class for all validators"""

    def __init__(self, threshold: float = None, **kwargs):
        self.threshold = threshold
        self.config = kwargs

    @abstractmethod
    def validate(
        self,
        output_data: pd.DataFrame,
        supporting_data: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> tuple[List[Violation], List[Warning]]:
        """
        Validate the data and return violations and warnings.

        Args:
            output_data: The output DataFrame being validated
            supporting_data: Supporting data for validation (e.g., counts, original values)
            metadata: Parsed metadata from description document

        Returns:
            Tuple of (violations, warnings)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the validator name"""
        pass

    def _is_percentage_column(
        self, column_name: str, column_data: pd.Series, metadata: DatasetMetadata
    ) -> tuple[bool, str]:
        """
        Determine if a column represents percentage/share data.

        Returns:
            Tuple of (is_percentage, reason)
        """
        reasons = []

        # Check metadata
        if column_name in metadata.percentage_columns:
            reasons.append("identified in metadata")

        # Check column name patterns
        percentage_keywords = [
            "percent", "percentage", "share", "proportion", "ratio", "rate", "%"
        ]
        if any(keyword in column_name.lower() for keyword in percentage_keywords):
            reasons.append("column name contains percentage keyword")

        # Check value ranges
        numeric_values = column_data.dropna()
        if len(numeric_values) > 0:
            # Check if values are in decimal percentage range (0-1)
            if numeric_values.between(0, 1).all():
                reasons.append("all values in [0, 1] range")
            # Check if values are in percentage format (0-100)
            elif numeric_values.between(0, 100).all() and numeric_values.max() > 1:
                reasons.append("all values in [0, 100] range")

        is_percentage = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "not detected as percentage"

        return is_percentage, reason_str