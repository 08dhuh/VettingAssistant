import io
from typing import Optional, List, Union

import pandas as pd

from .models import (
    RegressionValidationResult,
    RegressionMetadata,
    DatasetMetadata,
    Violation,
    Warning,
    ViolationType,
    WarningType,
)
from .parsers import DescriptionParser, RegressionParser, DataLoader


class RegressionChecker:
    """
    Orchestrates all validation checks for regression output datasets.

    This class coordinates the validation workflow:
    1. Parse description document to extract metadata (including variable types)
    2. Parse regression output file to extract coefficients and statistics
    3. Merge variable type information from description into regression metadata
    4. Run validation checks (degrees of freedom, R², numeric variable risk)
    5. Aggregate and return results
    """

    # Minimum degrees of freedom threshold
    MIN_DF_THRESHOLD = 10

    # R-squared thresholds
    R_SQUARED_VIOLATION_THRESHOLD = 0.9  # >= 0.9 with intercept is a violation
    R_SQUARED_WARNING_THRESHOLD = 0.1  # < 0.1 triggers a warning

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize RegressionChecker.

        Args:
            openai_api_key: OpenAI API key for description parsing
        """
        self.openai_api_key = openai_api_key

        # Initialize parsers
        self.description_parser = (
            DescriptionParser(api_key=openai_api_key) if openai_api_key else None
        )
        self.regression_parser = RegressionParser()
        self.data_loader = DataLoader()

    def validate(
        self,
        description: Union[str, DatasetMetadata, io.BytesIO],
        output: Union[io.BytesIO, str],
        supporting: Optional[Union[pd.DataFrame, io.BytesIO, str]] = None,
        description_file_type: str = "docx",
        output_file_type: str = "xlsx",
        supporting_file_type: Optional[str] = None,
    ) -> RegressionValidationResult:
        """
        Main validation entry point for regression outputs.

        Args:
            description: Description document (file path, BytesIO, or DatasetMetadata)
            output: Regression output file (BytesIO or file path)
            supporting: Optional supporting data for numeric variable checks
            description_file_type: Type of description file ('txt', 'docx', 'pdf')
            output_file_type: Type of output file ('csv', 'xlsx')
            supporting_file_type: Type of supporting file ('csv', 'xlsx')

        Returns:
            RegressionValidationResult with violations, warnings, and metadata
        """
        all_violations: List[Violation] = []
        all_warnings: List[Warning] = []

        # Step 1: Parse description document
        description_metadata = self._parse_description(description, description_file_type)

        # Step 2: Parse regression output file
        try:
            regression_metadata = self.regression_parser.parse(output, output_file_type)
        except ValueError as e:
            # If parsing fails, return early with error
            all_warnings.append(
                Warning(
                    warning_type=WarningType.DATA_TYPE_MISMATCH,
                    message=f"Failed to parse regression output: {e}",
                    context={"error": str(e)},
                )
            )
            return RegressionValidationResult(
                passed=False,
                violations=all_violations,
                warnings=all_warnings,
                regression_metadata=None,
                description_metadata=description_metadata,
                metadata={"error": str(e)},
            )

        # Step 3: Merge variable types from description into regression coefficients
        self._merge_variable_types(regression_metadata, description_metadata)

        # Step 4: Load supporting data if provided
        supporting_df = None
        if supporting is not None and supporting_file_type is not None:
            supporting_df = self._load_dataframe(supporting, supporting_file_type)

        # Step 5: Run validators
        # 5a: Validate degrees of freedom
        df_violations, df_warnings = self._validate_degrees_of_freedom(regression_metadata)
        all_violations.extend(df_violations)
        all_warnings.extend(df_warnings)

        # 5b: Validate R-squared (only for OLS)
        if regression_metadata.regression_type == "OLS":
            r2_violations, r2_warnings = self._validate_r_squared(regression_metadata)
            all_violations.extend(r2_violations)
            all_warnings.extend(r2_warnings)

        # 5c: Validate numeric variable disclosure risk (only for OLS with supporting data)
        rule_of_10_results = []
        if regression_metadata.regression_type == "OLS":
            nv_violations, nv_warnings, rule_of_10_results = self._validate_numeric_variable(
                regression_metadata, description_metadata, supporting_df
            )
            all_violations.extend(nv_violations)
            all_warnings.extend(nv_warnings)

        # Step 6: Check for variable mismatches between description and output
        mismatch_warnings = self._check_variable_mismatches(
            regression_metadata, description_metadata
        )
        all_warnings.extend(mismatch_warnings)

        # Step 7: Check metadata completeness
        if not description_metadata.is_complete():
            all_warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message="Description document is missing some required information.",
                    context={
                        "has_population": description_metadata.population is not None,
                        "has_method": description_metadata.method_of_analysis is not None,
                        "has_datasets": len(description_metadata.datasets_used or []) > 0,
                        "has_description": description_metadata.data_description is not None,
                        "has_variables": len(description_metadata.variables) > 0,
                    },
                )
            )

        # Step 8: Build and return result
        return RegressionValidationResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
            regression_metadata=regression_metadata,
            description_metadata=description_metadata,
            metadata={
                "n_observations": regression_metadata.n_observations,
                "r_squared": regression_metadata.r_squared,
                "degrees_of_freedom": regression_metadata.degrees_of_freedom,
                "n_independent_vars": regression_metadata.n_independent_vars,
                "has_intercept": regression_metadata.has_intercept,
                "has_continuous_variable": regression_metadata.has_continuous_variable,
                "rule_of_10_results": rule_of_10_results,
                "thresholds": {
                    "min_df": self.MIN_DF_THRESHOLD,
                    "r_squared_warning": self.R_SQUARED_WARNING_THRESHOLD,
                },
            },
        )

    def _parse_description(
        self,
        description: Union[str, DatasetMetadata, io.BytesIO],
        file_type: str,
    ) -> DatasetMetadata:
        """Parse description document or return existing metadata."""
        if isinstance(description, DatasetMetadata):
            return description

        if not self.description_parser:
            # If no OpenAI key provided, return minimal metadata
            if isinstance(description, str):
                return DatasetMetadata(raw_description=description)
            else:
                return DatasetMetadata(raw_description="[Binary content - parser not available]")

        # Parse with OpenAI
        return self.description_parser.parse(description, file_type)

    def _load_dataframe(
        self,
        data: Union[pd.DataFrame, str, io.BytesIO],
        file_type: str,
    ) -> pd.DataFrame:
        """Load DataFrame from various sources."""
        if isinstance(data, pd.DataFrame):
            return data

        return self.data_loader.load_dataframe(data, file_type)

    def _merge_variable_types(
        self,
        regression_metadata: RegressionMetadata,
        description_metadata: DatasetMetadata,
    ) -> None:
        """
        Merge variable type information from description into regression coefficients.

        Modifies regression_metadata.coefficients in place.
        """
        variable_types = description_metadata.variable_types

        for coef in regression_metadata.coefficients:
            # Skip if already has a type assigned
            if coef.variable_type is not None:
                continue

            # Look up in description variable types (case-insensitive)
            var_name_lower = coef.variable_name.lower()
            for desc_var, var_type in variable_types.items():
                if desc_var.lower() == var_name_lower:
                    coef.variable_type = var_type
                    break

    def _validate_degrees_of_freedom(
        self, regression_metadata: RegressionMetadata
    ) -> tuple[List[Violation], List[Warning]]:
        """
        Validate that degrees of freedom meets minimum threshold.

        For regression outputs, df = n - k - 1 where:
        - n = number of observations
        - k = number of independent variables
        """
        violations = []
        warnings = []

        df = regression_metadata.degrees_of_freedom

        if df < self.MIN_DF_THRESHOLD:
            violations.append(
                Violation(
                    violation_type=ViolationType.INSUFFICIENT_DF,
                    row=0,
                    column="degrees_of_freedom",
                    value=df,
                    context={
                        "n_observations": regression_metadata.n_observations,
                        "n_independent_vars": regression_metadata.n_independent_vars,
                        "threshold": self.MIN_DF_THRESHOLD,
                        "message": f"Degrees of freedom ({df}) is below minimum threshold ({self.MIN_DF_THRESHOLD})",
                    },
                )
            )

        return violations, warnings

    def _validate_r_squared(
        self, regression_metadata: RegressionMetadata
    ) -> tuple[List[Violation], List[Warning]]:
        """
        Validate R-squared value for OLS regression.

        - R² >= 0.9 with intercept present is a VIOLATION (disclosure risk)
        - Very low R-squared may indicate model issues (warning only)
        """
        violations = []
        warnings = []

        r2 = regression_metadata.r_squared

        # Check for high R² with intercept (violation)
        if r2 >= self.R_SQUARED_VIOLATION_THRESHOLD and regression_metadata.has_intercept:
            violations.append(
                Violation(
                    violation_type=ViolationType.HIGH_R_SQUARED,
                    row=0,
                    column="r_squared",
                    value=r2,
                    context={
                        "r_squared": r2,
                        "threshold": self.R_SQUARED_VIOLATION_THRESHOLD,
                        "has_intercept": True,
                        "message": f"R² ({r2:.4f}) ≥ {self.R_SQUARED_VIOLATION_THRESHOLD} with intercept present. "
                                   f"Either suppress the intercept or reduce R².",
                    },
                )
            )

        # Check for very low R² (warning only)
        if r2 < self.R_SQUARED_WARNING_THRESHOLD:
            warnings.append(
                Warning(
                    warning_type=WarningType.LOW_R_SQUARED,
                    message=f"R-squared ({r2:.4f}) is very low. Model may have poor explanatory power.",
                    context={
                        "r_squared": r2,
                        "threshold": self.R_SQUARED_WARNING_THRESHOLD,
                    },
                )
            )

        return violations, warnings

    # Minimum count threshold for Rule of 10
    MIN_COUNT_THRESHOLD = 10

    def _validate_numeric_variable(
        self,
        regression_metadata: RegressionMetadata,
        description_metadata: DatasetMetadata,
        supporting_df: Optional[pd.DataFrame],
    ) -> tuple[List[Violation], List[Warning], list]:
        """
        Validate numeric variable disclosure risk for OLS regression.

        Rules:
        1. If at least one continuous variable exists → check PASSES
        2. If NO continuous variables (all categorical/binary):
           - Supporting data (crosstab) MUST be provided
           - Each variable's count in supporting data must pass Rule of 10 (≥ 10)
           - If ANY variable fails Rule of 10 → check FAILS

        Returns:
            Tuple of (violations, warnings, rule_of_10_results)
        """
        violations = []
        warnings = []
        rule_of_10_results = []

        # Intercept names to exclude from variable checks
        intercept_names = {"constant", "intercept", "_cons", "(intercept)"}

        # Get independent variables (excluding intercept)
        independent_vars = [
            coef for coef in regression_metadata.coefficients
            if coef.variable_name.lower() not in intercept_names
        ]

        # Check if there are any continuous variables
        continuous_vars = [
            coef for coef in independent_vars
            if coef.variable_type == "continuous"
        ]

        # Check for unknown variable types
        unknown_vars = [
            coef for coef in independent_vars
            if coef.variable_type is None
        ]

        if unknown_vars:
            warnings.append(
                Warning(
                    warning_type=WarningType.VARIABLE_TYPE_UNKNOWN,
                    message=f"Could not determine variable type for: {', '.join(v.variable_name for v in unknown_vars)}",
                    context={
                        "unknown_variables": [v.variable_name for v in unknown_vars],
                    },
                )
            )

        # If continuous variables exist, the check passes (with low DF warning)
        if continuous_vars:
            if regression_metadata.degrees_of_freedom < 20:
                warnings.append(
                    Warning(
                        warning_type=WarningType.LOW_DF_WITH_CONTINUOUS,
                        message=f"Regression includes continuous variables with relatively low degrees of freedom ({regression_metadata.degrees_of_freedom}).",
                        context={
                            "continuous_variables": [v.variable_name for v in continuous_vars],
                            "degrees_of_freedom": regression_metadata.degrees_of_freedom,
                        },
                    )
                )
            return violations, warnings, rule_of_10_results

        # No continuous variables - all are categorical/binary
        # Supporting data (crosstab) MUST be provided and Rule of 10 must pass

        categorical_vars = [
            coef for coef in independent_vars
            if coef.variable_type in ("binary", "categorical")
        ]

        # If no continuous variables and we have categorical ones, need supporting data
        if categorical_vars or unknown_vars:
            if supporting_df is None:
                violations.append(
                    Violation(
                        violation_type=ViolationType.NUMERIC_VARIABLE,
                        row=0,
                        column="supporting_data",
                        value=None,
                        context={
                            "message": "All variables are categorical/binary. Supporting data (crosstab) is required "
                                       "to verify Rule of 10 for each variable.",
                            "categorical_variables": [v.variable_name for v in categorical_vars],
                            "unknown_variables": [v.variable_name for v in unknown_vars],
                        },
                    )
                )
                return violations, warnings, rule_of_10_results

            # Supporting data provided - check Rule of 10 for each variable
            rule_of_10_results = self._check_rule_of_10(
                independent_vars, supporting_df, intercept_names
            )

            # Add violations for variables that fail Rule of 10
            for var_result in rule_of_10_results:
                if var_result["status"] == "fail":
                    violations.append(
                        Violation(
                            violation_type=ViolationType.RULE_OF_10,
                            row=var_result.get("row", 0),
                            column=var_result["variable"],
                            value=var_result["count"],
                            context={
                                "variable": var_result["variable"],
                                "count": var_result["count"],
                                "threshold": self.MIN_COUNT_THRESHOLD,
                                "message": f"Variable '{var_result['variable']}' has count {var_result['count']} "
                                           f"which is below the minimum threshold of {self.MIN_COUNT_THRESHOLD}.",
                            },
                        )
                    )
                elif var_result["status"] == "not_found":
                    warnings.append(
                        Warning(
                            warning_type=WarningType.VARIABLE_MISMATCH,
                            message=f"Variable '{var_result['variable']}' from regression output not found in supporting data.",
                            context={
                                "variable": var_result["variable"],
                                "direction": "regression_only",
                            },
                        )
                    )

        return violations, warnings, rule_of_10_results

    def _check_rule_of_10(
        self,
        variables: list,
        supporting_df: pd.DataFrame,
        intercept_names: set,
    ) -> list:
        """
        Check Rule of 10 for each variable in supporting data.

        Args:
            variables: List of RegressionCoefficient objects
            supporting_df: DataFrame with variable names and counts
            intercept_names: Set of names to identify intercept

        Returns:
            List of dicts with check results for each variable
        """
        results = []

        # Normalize supporting data column names
        supporting_cols_lower = {col.lower(): col for col in supporting_df.columns}

        # Find the variable name column and count column
        var_col = None
        count_col = None

        # Common variable column names
        var_col_names = {"variable", "var", "name", "variable_name", "varname"}
        count_col_names = {"count", "n", "obs", "observations", "frequency", "freq"}

        for col_lower, col_orig in supporting_cols_lower.items():
            if col_lower in var_col_names:
                var_col = col_orig
            if col_lower in count_col_names:
                count_col = col_orig

        # If we couldn't identify columns, try to infer from data types
        if var_col is None or count_col is None:
            for col in supporting_df.columns:
                if supporting_df[col].dtype == 'object' and var_col is None:
                    var_col = col
                elif supporting_df[col].dtype in ('int64', 'float64') and count_col is None:
                    count_col = col

        if var_col is None or count_col is None:
            # Can't parse supporting data - return warning for all variables
            for var in variables:
                if var.variable_name.lower() not in intercept_names:
                    results.append({
                        "variable": var.variable_name,
                        "status": "not_found",
                        "count": None,
                    })
            return results

        # Build lookup from supporting data (case-insensitive)
        supporting_lookup = {}
        for idx, row in supporting_df.iterrows():
            var_name = str(row[var_col]).lower()
            count = row[count_col]
            supporting_lookup[var_name] = {"count": count, "row": idx}

        # Check each variable
        for var in variables:
            var_name = var.variable_name
            var_name_lower = var_name.lower()

            if var_name_lower in intercept_names:
                continue

            if var_name_lower in supporting_lookup:
                count = supporting_lookup[var_name_lower]["count"]
                row_idx = supporting_lookup[var_name_lower]["row"]

                if count >= self.MIN_COUNT_THRESHOLD:
                    results.append({
                        "variable": var_name,
                        "status": "pass",
                        "count": count,
                        "row": row_idx,
                    })
                else:
                    results.append({
                        "variable": var_name,
                        "status": "fail",
                        "count": count,
                        "row": row_idx,
                    })
            else:
                results.append({
                    "variable": var_name,
                    "status": "not_found",
                    "count": None,
                })

        return results

    def _check_variable_mismatches(
        self,
        regression_metadata: RegressionMetadata,
        description_metadata: DatasetMetadata,
    ) -> List[Warning]:
        """
        Check for mismatches between variables in description and regression output.
        """
        warnings = []

        # Get variable names from description (case-insensitive)
        desc_vars = {v.lower() for v in description_metadata.variables.keys()}

        # Get coefficient names from regression (excluding intercept)
        intercept_names = {"constant", "intercept", "_cons", "(intercept)"}
        reg_vars = {
            coef.variable_name.lower()
            for coef in regression_metadata.coefficients
            if coef.variable_name.lower() not in intercept_names
        }

        # Find variables in description but not in regression
        missing_from_reg = desc_vars - reg_vars
        if missing_from_reg:
            warnings.append(
                Warning(
                    warning_type=WarningType.VARIABLE_MISMATCH,
                    message=f"Variables in description but not in regression output: {', '.join(missing_from_reg)}",
                    context={
                        "missing_variables": list(missing_from_reg),
                        "direction": "description_only",
                    },
                )
            )

        # Find variables in regression but not in description
        missing_from_desc = reg_vars - desc_vars
        if missing_from_desc:
            warnings.append(
                Warning(
                    warning_type=WarningType.VARIABLE_MISMATCH,
                    message=f"Variables in regression output but not in description: {', '.join(missing_from_desc)}",
                    context={
                        "missing_variables": list(missing_from_desc),
                        "direction": "regression_only",
                    },
                )
            )

        return warnings
