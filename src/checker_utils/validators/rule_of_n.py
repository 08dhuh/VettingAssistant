from typing import List
import pandas as pd
import numpy as np
from .base_validator import BaseValidator
from ..models import Violation, Warning, ViolationType, DatasetMetadata


class RuleOfNValidator(BaseValidator):
    """
    Validates that all numeric values in supporting data meet minimum threshold (rule of N).

    Default: Rule of 10 - all sample sizes/counts must be >= 10
    """

    def __init__(self, threshold: int = 10):
        super().__init__(threshold=threshold)
        self.min_n = int(threshold)

    def get_name(self) -> str:
        return f"Rule of {self.min_n}"

    def validate(
        self,
        output_data: pd.DataFrame,
        supporting_data: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> tuple[List[Violation], List[Warning]]:
        """
        Check that all numeric cells in supporting_data are >= min_n threshold.

        Args:
            output_data: Not used for this validator (output means don't need rule of N check)
            supporting_data: DataFrame containing counts/sample sizes to validate
            metadata: Metadata for context

        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []

        # If sample_size_column is specified in metadata, only check that column
        if metadata.sample_size_column and metadata.sample_size_column in supporting_data.columns:
            columns_to_check = [metadata.sample_size_column]
        else:
            # Check all numeric columns
            columns_to_check = supporting_data.select_dtypes(include=[np.number]).columns

        for col in columns_to_check:
            for row_idx, cell_value in enumerate(supporting_data[col]):
                # Skip NaN/None values (they're allowed)
                if pd.isna(cell_value):
                    continue

                # Convert numpy types to Python types
                if isinstance(cell_value, (np.int64, np.float64)):
                    cell_value = float(cell_value)

                # Only check numeric values
                if isinstance(cell_value, (int, float)):
                    if cell_value < self.min_n:
                        violations.append(
                            Violation(
                                violation_type=ViolationType.RULE_OF_N,
                                row=row_idx,
                                column=col,
                                value=cell_value,
                                context={
                                    "threshold": self.min_n,
                                    "position": [row_idx, supporting_data.columns.get_loc(col)],
                                    "message": f"Value {cell_value} is below minimum threshold of {self.min_n}"
                                }
                            )
                        )

        return violations, warnings