import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import re


class DataCleaner:
    """
    Handles messy real-world data by cleaning and validating table structure.

    Addresses issues like:
    - Extraneous values outside table boundaries
    - Inconsistent row counts
    - Mixed data types
    - Summary rows/calculations at bottom
    """

    @staticmethod
    def clean_supporting_data(
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Clean supporting data by removing extraneous rows and validating structure.

        Args:
            df: Supporting DataFrame to clean
            reference_df: Optional output/means DataFrame to align with

        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()

        # Step 1: Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')

        # Step 2: Identify and remove summary/calculation rows
        cleaned_df = DataCleaner._remove_summary_rows(cleaned_df)

        # Step 3: If reference_df provided, align row counts
        if reference_df is not None:
            cleaned_df = DataCleaner._align_with_reference(cleaned_df, reference_df)

        # Step 4: Validate numeric columns
        cleaned_df = DataCleaner._clean_numeric_columns(cleaned_df)

        # Step 5: Remove rows with too many NaN values (likely extraneous)
        threshold = 0.5  # If more than 50% of values are NaN, drop the row
        cleaned_df = cleaned_df.dropna(thresh=int(len(cleaned_df.columns) * threshold))

        # Reset index after cleaning
        cleaned_df = cleaned_df.reset_index(drop=True)

        return cleaned_df

    @staticmethod
    def _remove_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that appear to be summaries, totals, or calculations.

        Indicators:
        - Rows with values much larger than others (aggregates)
        - Rows with calculated values (ratios like 0.758...)
        - Rows at the end with only 1-2 values
        """
        if len(df) == 0:
            return df

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return df

        # Strategy 1: Remove rows with decimal values close to 0-1 range
        # (likely calculated ratios/percentages not part of main data)
        rows_to_keep = []

        for idx, row in df.iterrows():
            numeric_values = [v for v in row[numeric_cols] if pd.notna(v)]

            if not numeric_values:
                rows_to_keep.append(True)
                continue

            # Check if this looks like a ratio/percentage (between 0 and 1)
            # but NOT if all values in the column are in this range
            is_likely_ratio = False
            for col in numeric_cols:
                val = row[col]
                if pd.notna(val):
                    # Is this a decimal ratio while most column values are large?
                    if 0 < val < 1:
                        col_median = df[col].median()
                        if col_median > 10:  # Column has large values
                            is_likely_ratio = True
                            break

            rows_to_keep.append(not is_likely_ratio)

        df_filtered = df[rows_to_keep].copy()

        # Strategy 2: Remove rows where values are outliers (e.g., 100x larger than median)
        # This catches summary totals
        for col in numeric_cols:
            if df_filtered[col].notna().sum() < 3:
                continue

            col_median = df_filtered[col].median()
            if col_median == 0:
                continue

            # Flag values that are more than 100x the median as potential summaries
            outlier_threshold = 100
            outliers = df_filtered[col] > (col_median * outlier_threshold)

            # Only remove if it's in the last 20% of rows (summaries usually at bottom)
            last_20_pct = int(len(df_filtered) * 0.8)
            for idx in df_filtered[outliers].index:
                if idx >= last_20_pct:
                    df_filtered = df_filtered.drop(idx)

        return df_filtered.reset_index(drop=True)

    @staticmethod
    def _align_with_reference(
        supporting_df: pd.DataFrame,
        reference_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align supporting data with reference DataFrame (e.g., output means).

        Strategy:
        - Match by common index column (e.g., 'year')
        - Or ensure row count matches
        - Remove extra rows that don't align
        """
        # Try to find common index column
        common_cols = set(supporting_df.columns).intersection(set(reference_df.columns))

        # Look for typical index columns
        index_candidates = ['year', 'id', 'index', 'group']
        index_col = None

        for candidate in index_candidates:
            if candidate in common_cols:
                index_col = candidate
                break

        if index_col:
            # Align by index column
            reference_values = set(reference_df[index_col].dropna())
            supporting_values = set(supporting_df[index_col].dropna())

            # Keep only rows that match reference
            aligned_df = supporting_df[supporting_df[index_col].isin(reference_values)].copy()
            return aligned_df.reset_index(drop=True)

        else:
            # No common index - just ensure row count matches
            # Take first N rows where N = len(reference_df)
            target_rows = len(reference_df)
            if len(supporting_df) > target_rows:
                return supporting_df.head(target_rows).copy()

        return supporting_df

    @staticmethod
    def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric columns by removing non-numeric characters and converting types.
        """
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Try to clean and convert to numeric
            try:
                # Remove currency symbols, commas, etc.
                cleaned = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
                cleaned = cleaned.replace('nan', np.nan)
                df[col] = pd.to_numeric(cleaned, errors='coerce')
            except:
                # If conversion fails, leave as is
                pass

        return df

    @staticmethod
    def validate_table_structure(
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        min_rows: int = 1
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the DataFrame has expected structure.

        Args:
            df: DataFrame to validate
            expected_columns: List of required column names (optional)
            min_rows: Minimum number of rows required

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check row count
        if len(df) < min_rows:
            issues.append(f"Too few rows: {len(df)} (minimum: {min_rows})")

        # Check for expected columns
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")

        # Check for all-NaN columns
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            issues.append(f"Columns with all NaN values: {all_nan_cols}")

        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            issues.append(f"Duplicate column names: {duplicate_cols}")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def detect_and_remove_header_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove duplicate header rows that sometimes appear in messy data.
        """
        if len(df) < 2:
            return df

        # Check if first row(s) contain the same values as column names
        rows_to_drop = []

        for idx in range(min(3, len(df))):  # Check first 3 rows
            row_values = df.iloc[idx].astype(str).str.lower().str.strip().tolist()
            col_names = df.columns.astype(str).str.lower().str.strip().tolist()

            # If row values match column names, it's a duplicate header
            matches = sum(1 for rv, cn in zip(row_values, col_names) if rv == cn)
            if matches / len(col_names) > 0.5:  # More than 50% match
                rows_to_drop.append(idx)

        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)

        return df
