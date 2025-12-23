import io
import re
from typing import Tuple, Union, Optional, List
from pathlib import Path

import pandas as pd

from ..models import RegressionMetadata, RegressionCoefficient


class RegressionParser:
    """
    Parses regression output files (Excel/CSV) to extract coefficients and metadata.
    Supports standard regression output formats with coefficients, standard errors,
    and significance indicators.
    """

    # Patterns for identifying metadata rows (not coefficients)
    # These must match the full variable name (exact match)
    _N_OBS_EXACT_PATTERNS = {"n", "obs", "nobs", "observations", "sample size"}
    # These can be substring matches
    _N_OBS_SUBSTRING_PATTERNS = {"number of obs", "num obs", "n obs"}
    _R_SQUARED_PATTERNS = {"r-squared", "r2", "r-sq", "rsquared", "r²", "r squared"}

    # Patterns for identifying intercept
    _INTERCEPT_PATTERNS = {"constant", "intercept", "_cons", "(intercept)"}

    def parse(
        self,
        file_content: Union[str, bytes, io.BytesIO, Path],
        file_type: str = "xlsx"
    ) -> RegressionMetadata:
        """
        Parse regression output file and extract metadata.

        Args:
            file_content: File path (str/Path) or file content (bytes/BytesIO)
            file_type: File type ('csv' or 'xlsx')

        Returns:
            RegressionMetadata object with extracted coefficients and statistics
        """
        df = self._load_dataframe(file_content, file_type)
        return self._parse_dataframe(df)

    def _load_dataframe(
        self,
        file_content: Union[str, bytes, io.BytesIO, Path],
        file_type: str
    ) -> pd.DataFrame:
        """Load DataFrame from file content."""
        if file_type == "csv":
            if isinstance(file_content, (str, Path)):
                return pd.read_csv(file_content)
            elif isinstance(file_content, bytes):
                return pd.read_csv(io.BytesIO(file_content))
            elif isinstance(file_content, io.BytesIO):
                return pd.read_csv(file_content)
            else:
                raise ValueError(f"Unsupported file_content type: {type(file_content)}")

        elif file_type in ("xlsx", "xls"):
            if isinstance(file_content, (str, Path)):
                return pd.read_excel(file_content)
            elif isinstance(file_content, bytes):
                return pd.read_excel(io.BytesIO(file_content))
            elif isinstance(file_content, io.BytesIO):
                return pd.read_excel(file_content)
            else:
                raise ValueError(f"Unsupported file_content type: {type(file_content)}")

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _parse_dataframe(self, df: pd.DataFrame) -> RegressionMetadata:
        """Parse DataFrame to extract regression metadata."""
        # Identify columns
        var_col = self._find_variable_column(df)
        coef_col = self._find_coefficient_column(df)
        std_err_col = self._find_std_error_column(df)

        if var_col is None:
            raise ValueError("Could not identify variable column in regression output")
        if coef_col is None:
            raise ValueError("Could not identify coefficient column in regression output")

        # Extract metadata and coefficients
        n_observations: Optional[int] = None
        r_squared: Optional[float] = None
        coefficients: List[RegressionCoefficient] = []
        has_intercept = False

        for _, row in df.iterrows():
            var_name = str(row[var_col]).strip() if pd.notna(row[var_col]) else ""
            var_name_lower = var_name.lower()

            # Check for metadata rows
            if self._is_n_observations_row(var_name_lower):
                n_observations = self._extract_n_observations(row, coef_col, std_err_col)
                continue

            if self._is_r_squared_row(var_name_lower):
                r_squared = self._extract_r_squared(row, coef_col, std_err_col)
                continue

            # Skip empty rows or rows without valid variable names
            if not var_name or var_name_lower in ("", "nan"):
                continue

            # Parse coefficient row
            coef_value = row[coef_col] if pd.notna(row[coef_col]) else None
            if coef_value is None:
                continue

            coefficient, stars = self._parse_coefficient_value(coef_value)

            std_error = None
            if std_err_col is not None and pd.notna(row[std_err_col]):
                std_error = self._parse_numeric_value(row[std_err_col])

            # Check if this is intercept
            if var_name_lower in self._INTERCEPT_PATTERNS:
                has_intercept = True

            coefficients.append(RegressionCoefficient(
                variable_name=var_name,
                coefficient=coefficient,
                std_error=std_error,
                significance_level=stars,
                variable_type=None,  # Cannot determine from regression output alone
            ))

        # Validate required fields
        if n_observations is None:
            raise ValueError("Could not find number of observations in regression output")
        if r_squared is None:
            raise ValueError("Could not find R-squared in regression output")

        return RegressionMetadata(
            n_observations=n_observations,
            r_squared=r_squared,
            coefficients=coefficients,
            has_intercept=has_intercept,
            regression_type="OLS",
        )

    def _find_variable_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the column containing variable names."""
        # Check for explicit variable column
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ("variable", "var", "name", "variables"):
                return col

        # Default to first column
        if len(df.columns) > 0:
            return df.columns[0]

        return None

    def _find_coefficient_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the column containing coefficient values."""
        for col in df.columns:
            col_lower = str(col).lower()
            if "coef" in col_lower or "estimate" in col_lower or col_lower == "b":
                return col
        return None

    def _find_std_error_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the column containing standard errors."""
        for col in df.columns:
            col_lower = str(col).lower()
            if "std" in col_lower or "se" in col_lower or "error" in col_lower:
                return col
        return None

    def _is_n_observations_row(self, var_name_lower: str) -> bool:
        """Check if row contains number of observations."""
        # Check exact matches first
        if var_name_lower in self._N_OBS_EXACT_PATTERNS:
            return True
        # Check substring matches
        return any(pattern in var_name_lower for pattern in self._N_OBS_SUBSTRING_PATTERNS)

    def _is_r_squared_row(self, var_name_lower: str) -> bool:
        """Check if row contains R-squared value."""
        return any(pattern in var_name_lower for pattern in self._R_SQUARED_PATTERNS)

    def _extract_n_observations(self, row: pd.Series, coef_col: str, std_err_col: Optional[str]) -> int:
        """Extract number of observations from row."""
        # Try coefficient column first, then std error column
        for col in [coef_col, std_err_col]:
            if col is not None and pd.notna(row[col]):
                try:
                    return parse_integer_with_commas(row[col])
                except (ValueError, TypeError):
                    continue
        raise ValueError("Could not extract number of observations")

    def _extract_r_squared(self, row: pd.Series, coef_col: str, std_err_col: Optional[str]) -> float:
        """Extract R-squared value from row."""
        # Try coefficient column first, then std error column
        for col in [coef_col, std_err_col]:
            if col is not None and pd.notna(row[col]):
                try:
                    value = self._parse_numeric_value(row[col])
                    if value is not None:
                        return value
                except (ValueError, TypeError):
                    continue
        raise ValueError("Could not extract R-squared value")

    def _parse_coefficient_value(self, value) -> Tuple[float, int]:
        """
        Parse a coefficient value that may contain asterisks.

        Args:
            value: Coefficient value (e.g., "0.5124***" or 0.5124)

        Returns:
            Tuple of (coefficient_float, star_count)
        """
        return parse_coefficient_value(value)

    def _parse_numeric_value(self, value) -> Optional[float]:
        """Parse a numeric value, handling commas and string formats."""
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        # Handle string values
        str_value = str(value).strip()
        # Remove commas
        str_value = str_value.replace(",", "")
        # Remove any asterisks (shouldn't be in std error, but just in case)
        str_value = str_value.rstrip("*")

        try:
            return float(str_value)
        except ValueError:
            return None


def parse_coefficient_value(value) -> Tuple[float, int]:
    """
    Parse a coefficient value that may contain asterisks.

    Args:
        value: Coefficient value (e.g., "0.5124***" or 0.5124)

    Returns:
        Tuple of (coefficient_float, star_count)

    Examples:
        >>> parse_coefficient_value("0.5124***")
        (0.5124, 3)
        >>> parse_coefficient_value("0.277")
        (0.277, 0)
        >>> parse_coefficient_value(-1.234)
        (-1.234, 0)
    """
    if isinstance(value, (int, float)):
        return float(value), 0

    str_value = str(value).strip()

    # Count trailing asterisks
    star_count = 0
    while str_value.endswith("*"):
        star_count += 1
        str_value = str_value[:-1]

    # Remove commas and parse
    str_value = str_value.replace(",", "")

    try:
        coef = float(str_value)
    except ValueError:
        raise ValueError(f"Could not parse coefficient value: {value}")

    return coef, star_count


def parse_integer_with_commas(value) -> int:
    """
    Parse an integer value that may contain commas.

    Args:
        value: Integer value (e.g., "1,561,234" or 1561234)

    Returns:
        Integer value

    Examples:
        >>> parse_integer_with_commas("1,561,234")
        1561234
        >>> parse_integer_with_commas(1000)
        1000
    """
    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    str_value = str(value).strip()
    # Remove commas
    str_value = str_value.replace(",", "")

    try:
        # Handle float strings like "1561234.0"
        return int(float(str_value))
    except ValueError:
        raise ValueError(f"Could not parse integer value: {value}")
