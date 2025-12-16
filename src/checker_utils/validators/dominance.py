from typing import List, Optional
import pandas as pd
from .base_validator import BaseValidator
from ..models import (
    Violation,
    Warning,
    ViolationType,
    WarningType,
    DatasetMetadata,
)
from ..parsers.column_mapper import ColumnMapper


class DominanceD50Validator(BaseValidator):
    """
    Validates D50 dominance rule: Highest individual value must be <= 50% of total.

    Correct logic:
    - total_sum = mean (from output) × observations (from supporting)
    - For each row: check if highest_value / total_sum > 0.50
    """

    def __init__(self, threshold: float = 0.50, openai_api_key: Optional[str] = None):
        super().__init__(threshold=threshold)
        self.column_mapper = ColumnMapper(api_key=openai_api_key)
        self.all_checks = []  # Store all checks for summary display

    def get_name(self) -> str:
        return f"Dominance D50 (>{self.threshold*100:.0f}%)"

    def validate(
        self,
        output_data: pd.DataFrame,
        supporting_data: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> tuple[List[Violation], List[Warning]]:
        """
        Check if highest individual value exceeds threshold of (mean × observations).

        Args:
            output_data: DataFrame with means/averages
            supporting_data: DataFrame with highest values and observation counts
            metadata: Metadata for column identification

        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []
        self.all_checks = []  # Reset and track all D50 checks for summary display

        # Use LLM to intelligently map columns in supporting data
        supporting_mapping = self.column_mapper.map_columns(
            supporting_data,
            required_columns=["highest_value", "observations"],
            context="Supporting data for dominance checking. 'observations' is the count/number of data points, NOT dollar amounts."
        )

        highest_col = supporting_mapping.get("highest_value")
        obs_col = supporting_mapping.get("observations")

        if not highest_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'highest value' column in supporting data. D50 check requires this. Found columns: {list(supporting_data.columns)}",
                    context={
                        "required_column": "highest_value",
                        "available_columns": list(supporting_data.columns)
                    }
                )
            )
            return violations, warnings

        if not obs_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'observations' column in supporting data. D50 check requires this. Found columns: {list(supporting_data.columns)}",
                    context={
                        "required_column": "observations/count",
                        "available_columns": list(supporting_data.columns)
                    }
                )
            )
            return violations, warnings

        # Use LLM to intelligently map columns in output data
        output_mapping = self.column_mapper.map_columns(
            output_data,
            required_columns=["mean"],
            context="Output data containing calculated means/averages"
        )

        mean_col = output_mapping.get("mean")

        print(f"\n[D50 DEBUG] Output data columns: {list(output_data.columns)}")
        print(f"[D50 DEBUG] Supporting data columns: {list(supporting_data.columns)}")
        print(f"[D50 DEBUG] LLM-detected mean column: '{mean_col}'")
        print(f"[D50 DEBUG] LLM-detected highest column: '{highest_col}'")
        print(f"[D50 DEBUG] LLM-detected observations column: '{obs_col}'")

        if not mean_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'mean' column in output data. D50 check requires this. Found columns: {list(output_data.columns)}",
                    context={
                        "required_column": "mean/average",
                        "available_columns": list(output_data.columns)
                    }
                )
            )
            return violations, warnings

        # Align output and supporting data
        aligned_data = self._align_dataframes(output_data, supporting_data)

        if aligned_data is None:
            warnings.append(
                Warning(
                    warning_type=WarningType.DATA_TYPE_MISMATCH,
                    message="Could not align output and supporting data. Ensure they have matching rows or a common index column.",
                    context={}
                )
            )
            return violations, warnings

        # Debug logging
        print(f"\n[D50 DEBUG] Aligned data shape: {aligned_data.shape}")
        print(f"[D50 DEBUG] Mean column detected: '{mean_col}'")
        print(f"[D50 DEBUG] Highest column detected: '{highest_col}'")
        print(f"[D50 DEBUG] Observations column detected: '{obs_col}'")
        print(f"[D50 DEBUG] Aligned columns: {list(aligned_data.columns)}")
        print(f"[D50 DEBUG] First 3 rows:")
        print(aligned_data[[mean_col, obs_col, highest_col]].head(3))

        # Perform D50 check
        for idx, row in aligned_data.iterrows():
            mean_value = row[mean_col]
            observations = row[obs_col]
            highest_value = row[highest_col]

            # Skip if any value is NaN
            if pd.isna(mean_value) or pd.isna(observations) or pd.isna(highest_value):
                continue

            # Convert to float
            mean_value = float(mean_value)
            observations = float(observations)
            highest_value = float(highest_value)

            # Calculate total sum = mean × observations
            total_sum = mean_value * observations

            if total_sum == 0:
                continue

            # Calculate dominance ratio
            dominance_ratio = highest_value / total_sum

            # Debug logging for each row
            print(f"[D50 DEBUG] Row {idx}: mean={mean_value:,.2f}, obs={observations:,.0f}, "
                  f"total_sum={total_sum:,.2f}, highest={highest_value:,.2f}, "
                  f"ratio={dominance_ratio*100:.2f}% (threshold={self.threshold*100:.0f}%)")

            # Store check result for summary display
            self.all_checks.append({
                'row': int(idx),
                'mean': float(mean_value),
                'observations': float(observations),
                'total_sum': float(total_sum),
                'highest_value': float(highest_value),
                'd50_ratio': float(dominance_ratio),
            })

            if dominance_ratio > self.threshold:
                # Check if this is a percentage column
                is_pct, pct_reason = self._is_percentage_column(
                    highest_col, supporting_data[highest_col], metadata
                )

                violations.append(
                    Violation(
                        violation_type=ViolationType.DOMINANCE_D50,
                        row=int(idx),
                        column=highest_col,
                        value=float(highest_value),
                        context={
                            "mean": float(mean_value),
                            "observations": float(observations),
                            "total_sum": float(total_sum),
                            "dominance_ratio": float(dominance_ratio),
                            "threshold": self.threshold,
                            "is_percentage_column": is_pct,
                            "percentage_detection_reason": pct_reason,
                        }
                    )
                )

        return violations, warnings

    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find column matching any of the keywords (case-insensitive)"""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword.lower() in col_lower for keyword in keywords):
                return col
        return None

    def _find_mean_column(self, df: pd.DataFrame, metadata: DatasetMetadata) -> Optional[str]:
        """Find the column containing mean/average values"""
        # Try metadata first
        if metadata.variables:
            for var_name, description in metadata.variables.items():
                if var_name in df.columns:
                    desc_lower = description.lower()
                    if any(keyword in desc_lower for keyword in ["mean", "average", "avg"]):
                        return var_name

        # Try common column name patterns
        mean_keywords = ["mean", "average", "avg", "income", "value", "amount"]
        return self._find_column(df, mean_keywords)

    def _align_dataframes(
        self,
        output_df: pd.DataFrame,
        supporting_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Align output and supporting dataframes by row.

        Returns combined DataFrame with columns from both, or None if alignment fails.
        """
        # Check if they have the same number of rows
        if len(output_df) == len(supporting_df):
            # Simple index-based alignment
            combined = pd.concat([
                output_df.reset_index(drop=True),
                supporting_df.reset_index(drop=True)
            ], axis=1)
            return combined

        # Try to find common index column
        common_cols = set(output_df.columns).intersection(set(supporting_df.columns))
        index_candidates = ['year', 'id', 'index', 'group', 'period']

        for candidate in index_candidates:
            if candidate in common_cols:
                # Merge on this column
                combined = pd.merge(
                    output_df,
                    supporting_df,
                    on=candidate,
                    how='inner',
                    suffixes=('_output', '_supporting')
                )
                return combined

        # No alignment possible
        return None


class DominanceD67Validator(BaseValidator):
    """
    Validates D67 dominance rule: Sum of first + second highest must be <= 67% of total.

    Correct logic:
    - total_sum = mean (from output) × observations (from supporting)
    - For each row: check if (first_highest + second_highest) / total_sum > 0.67
    """

    def __init__(self, threshold: float = 0.67, openai_api_key: Optional[str] = None):
        super().__init__(threshold=threshold)
        self.column_mapper = ColumnMapper(api_key=openai_api_key)
        self.all_checks = []  # Store all checks for summary display

    def get_name(self) -> str:
        return f"Dominance D67 (>{self.threshold*100:.0f}%)"

    def validate(
        self,
        output_data: pd.DataFrame,
        supporting_data: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> tuple[List[Violation], List[Warning]]:
        """
        Check if sum of first + second highest exceeds threshold of (mean × observations).

        Args:
            output_data: DataFrame with means/averages
            supporting_data: DataFrame with first/second highest and observation counts
            metadata: Metadata for column identification

        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []
        self.all_checks = []  # Reset and track all D67 checks for summary display

        # Use LLM to intelligently map columns in supporting data
        supporting_mapping = self.column_mapper.map_columns(
            supporting_data,
            required_columns=["first_highest", "second_highest", "observations"],
            context="Supporting data for dominance checking. 'observations' is the count/number of data points, NOT dollar amounts."
        )

        first_col = supporting_mapping.get("first_highest")
        second_col = supporting_mapping.get("second_highest")
        obs_col = supporting_mapping.get("observations")

        if not first_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'first highest' column in supporting data. D67 check requires this. Found columns: {list(supporting_data.columns)}",
                    context={
                        "required_column": "first_highest",
                        "available_columns": list(supporting_data.columns)
                    }
                )
            )
            return violations, warnings

        if not second_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'second highest' column in supporting data. D67 check requires this. Found columns: {list(supporting_data.columns)}",
                    context={
                        "required_column": "second_highest",
                        "available_columns": list(supporting_data.columns)
                    }
                )
            )
            return violations, warnings

        if not obs_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'observations' column in supporting data. D67 check requires this. Found columns: {list(supporting_data.columns)}",
                    context={
                        "required_column": "observations/count",
                        "available_columns": list(supporting_data.columns)
                    }
                )
            )
            return violations, warnings

        # Use LLM to intelligently map columns in output data
        output_mapping = self.column_mapper.map_columns(
            output_data,
            required_columns=["mean"],
            context="Output data containing calculated means/averages"
        )

        mean_col = output_mapping.get("mean")

        print(f"\n[D67 DEBUG] Output data columns: {list(output_data.columns)}")
        print(f"[D67 DEBUG] Supporting data columns: {list(supporting_data.columns)}")
        print(f"[D67 DEBUG] LLM-detected mean column: '{mean_col}'")
        print(f"[D67 DEBUG] LLM-detected first column: '{first_col}'")
        print(f"[D67 DEBUG] LLM-detected second column: '{second_col}'")
        print(f"[D67 DEBUG] LLM-detected observations column: '{obs_col}'")

        if not mean_col:
            warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message=f"Could not find 'mean' column in output data. D67 check requires this. Found columns: {list(output_data.columns)}",
                    context={
                        "required_column": "mean/average",
                        "available_columns": list(output_data.columns)
                    }
                )
            )
            return violations, warnings

        # Align output and supporting data
        aligned_data = self._align_dataframes(output_data, supporting_data)

        if aligned_data is None:
            warnings.append(
                Warning(
                    warning_type=WarningType.DATA_TYPE_MISMATCH,
                    message="Could not align output and supporting data. Ensure they have matching rows or a common index column.",
                    context={}
                )
            )
            return violations, warnings

        # Debug logging
        print(f"\n[D67 DEBUG] Aligned data shape: {aligned_data.shape}")
        print(f"[D67 DEBUG] Mean column detected: '{mean_col}'")
        print(f"[D67 DEBUG] First column detected: '{first_col}'")
        print(f"[D67 DEBUG] Second column detected: '{second_col}'")
        print(f"[D67 DEBUG] Observations column detected: '{obs_col}'")
        print(f"[D67 DEBUG] Aligned columns: {list(aligned_data.columns)}")
        print(f"[D67 DEBUG] First 3 rows:")
        print(aligned_data[[mean_col, obs_col, first_col, second_col]].head(3))

        # Perform D67 check
        for idx, row in aligned_data.iterrows():
            mean_value = row[mean_col]
            observations = row[obs_col]
            first_value = row[first_col]
            second_value = row[second_col]

            # Skip if any value is NaN
            if pd.isna(mean_value) or pd.isna(observations) or pd.isna(first_value) or pd.isna(second_value):
                continue

            # Convert to float
            mean_value = float(mean_value)
            observations = float(observations)
            first_value = float(first_value)
            second_value = float(second_value)

            # Calculate total sum = mean × observations
            total_sum = mean_value * observations

            if total_sum == 0:
                continue

            # Calculate D67 sum and ratio
            d67_sum = first_value + second_value
            d67_ratio = d67_sum / total_sum

            # Debug logging for each row
            print(f"[D67 DEBUG] Row {idx}: mean={mean_value:,.2f}, obs={observations:,.0f}, "
                  f"total_sum={total_sum:,.2f}, first={first_value:,.2f}, second={second_value:,.2f}, "
                  f"d67_sum={d67_sum:,.2f}, ratio={d67_ratio*100:.2f}% (threshold={self.threshold*100:.0f}%)")

            # Store check result for summary display
            self.all_checks.append({
                'row': int(idx),
                'mean': float(mean_value),
                'observations': float(observations),
                'total_sum': float(total_sum),
                'first_value': float(first_value),
                'second_value': float(second_value),
                'd67_sum': float(d67_sum),
                'd67_ratio': float(d67_ratio),
            })

            if d67_ratio > self.threshold:
                # Check if these are percentage columns
                is_pct_first, _ = self._is_percentage_column(first_col, supporting_data[first_col], metadata)
                is_pct_second, _ = self._is_percentage_column(second_col, supporting_data[second_col], metadata)

                violations.append(
                    Violation(
                        violation_type=ViolationType.DOMINANCE_D67,
                        row=int(idx),
                        column=f"{first_col}+{second_col}",
                        value=float(d67_sum),
                        context={
                            "mean": float(mean_value),
                            "observations": float(observations),
                            "total_sum": float(total_sum),
                            "first_value": float(first_value),
                            "second_value": float(second_value),
                            "d67_ratio": float(d67_ratio),
                            "threshold": self.threshold,
                            "first_column": first_col,
                            "second_column": second_col,
                            "is_percentage_column": is_pct_first and is_pct_second,
                        }
                    )
                )

        return violations, warnings

    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find column matching any of the keywords (case-insensitive)"""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword.lower() in col_lower for keyword in keywords):
                return col
        return None

    def _find_mean_column(self, df: pd.DataFrame, metadata: DatasetMetadata) -> Optional[str]:
        """Find the column containing mean/average values"""
        # Try metadata first
        if metadata.variables:
            for var_name, description in metadata.variables.items():
                if var_name in df.columns:
                    desc_lower = description.lower()
                    if any(keyword in desc_lower for keyword in ["mean", "average", "avg"]):
                        return var_name

        # Try common column name patterns
        mean_keywords = ["mean", "average", "avg", "income", "value", "amount"]
        return self._find_column(df, mean_keywords)

    def _align_dataframes(
        self,
        output_df: pd.DataFrame,
        supporting_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Align output and supporting dataframes by row.

        Returns combined DataFrame with columns from both, or None if alignment fails.
        """
        # Check if they have the same number of rows
        if len(output_df) == len(supporting_df):
            # Simple index-based alignment
            combined = pd.concat([
                output_df.reset_index(drop=True),
                supporting_df.reset_index(drop=True)
            ], axis=1)
            return combined

        # Try to find common index column
        common_cols = set(output_df.columns).intersection(set(supporting_df.columns))
        index_candidates = ['year', 'id', 'index', 'group', 'period']

        for candidate in index_candidates:
            if candidate in common_cols:
                # Merge on this column
                combined = pd.merge(
                    output_df,
                    supporting_df,
                    on=candidate,
                    how='inner',
                    suffixes=('_output', '_supporting')
                )
                return combined

        # No alignment possible
        return None
