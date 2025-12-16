import numpy as np
import pandas as pd
from typing import Union, Optional, List
import io

from .models import ValidationResult, Violation, Warning, DatasetMetadata, WarningType
from .validators import RuleOfNValidator, DominanceD50Validator, DominanceD67Validator
from .parsers import DescriptionParser, DataLoader, DataCleaner


class MeansChecker:
    """
    Orchestrates all validation checks for means/averages datasets.

    This class coordinates the validation workflow:
    1. Parse description document to extract metadata
    2. Load output and supporting data
    3. Run validation checks (Rule of N, Dominance D50, Dominance D67)
    4. Aggregate and return results
    """

    def __init__(
        self,
        min_n_threshold: int = 10,
        d50_threshold: float = 0.50,
        d67_threshold: float = 0.67,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize MeansChecker with validation thresholds.

        Args:
            min_n_threshold: Minimum sample size (default: 10)
            d50_threshold: D50 dominance threshold (default: 0.50)
            d67_threshold: D67 dominance threshold (default: 0.67)
            openai_api_key: OpenAI API key for description parsing
        """
        self.config = {
            "min_n": min_n_threshold,
            "d50": d50_threshold,
            "d67": d67_threshold,
        }
        self.openai_api_key = openai_api_key

        # Initialize validators (pass API key for intelligent column mapping)
        self.validators = {
            "rule_of_n": RuleOfNValidator(threshold=min_n_threshold),
            "dominance_d50": DominanceD50Validator(threshold=d50_threshold, openai_api_key=openai_api_key),
            "dominance_d67": DominanceD67Validator(threshold=d67_threshold, openai_api_key=openai_api_key),
        }

        # Initialize parsers
        self.description_parser = DescriptionParser(api_key=openai_api_key) if openai_api_key else None
        self.data_loader = DataLoader()

    def validate(
        self,
        description: Union[str, DatasetMetadata, io.BytesIO],
        output: Union[pd.DataFrame, str, io.BytesIO],
        supporting: Union[pd.DataFrame, str, io.BytesIO],
        description_file_type: str = "txt",
        output_file_type: str = "csv",
        supporting_file_type: str = "csv",
    ) -> ValidationResult:
        """
        Main validation entry point.

        Args:
            description: Description document (file path, BytesIO, or DatasetMetadata)
            output: Output means DataFrame or file
            supporting: Supporting data DataFrame or file
            description_file_type: Type of description file ('txt', 'docx', 'pdf')
            output_file_type: Type of output file ('csv', 'xlsx')
            supporting_file_type: Type of supporting file ('csv', 'xlsx')

        Returns:
            ValidationResult with violations, warnings, and metadata
        """
        # Step 1: Parse description
        metadata = self._parse_description(description, description_file_type)

        # Step 2: Load data
        output_df = self._load_dataframe(output, output_file_type)
        supporting_df_raw = self._load_dataframe(supporting, supporting_file_type)

        # Step 2.5: Clean supporting data to remove extraneous rows/values
        supporting_df = DataCleaner.clean_supporting_data(
            supporting_df_raw,
            reference_df=output_df
        )

        # Initialize violation and warning lists
        all_violations: List[Violation] = []
        all_warnings: List[Warning] = []

        # Validate table structure
        is_valid, issues = DataCleaner.validate_table_structure(supporting_df, min_rows=1)
        if not is_valid:
            # Add warnings for validation issues
            for issue in issues:
                all_warnings.append(
                    Warning(
                        warning_type=WarningType.DATA_TYPE_MISMATCH,
                        message=f"Supporting data structure issue: {issue}",
                        context={"validation_issue": issue}
                    )
                )

        # Step 3: Extract relevant columns from output
        # (For means data, we typically don't validate the output itself,
        # but the supporting data that shows counts/individual values)
        relevant_output = self._extract_relevant_columns(output_df, metadata)

        # Step 4: Run all validators

        for validator in self.validators.values():
            violations, warnings = validator.validate(
                output_data=relevant_output,
                supporting_data=supporting_df,
                metadata=metadata,
            )
            all_violations.extend(violations)
            all_warnings.extend(warnings)

        # Step 5: Check metadata completeness
        if not metadata.is_complete():
            all_warnings.append(
                Warning(
                    warning_type=WarningType.METADATA_INCOMPLETE,
                    message="Description document is missing some required information. This may affect validation accuracy.",
                    context={
                        "has_population": metadata.population is not None,
                        "has_method": metadata.method_of_analysis is not None,
                        "has_datasets": len(metadata.datasets_used) > 0,
                        "has_description": metadata.data_description is not None,
                        "has_variables": len(metadata.variables) > 0,
                    }
                )
            )

        # Step 6: Collect all dominance checks for summary display
        all_d50_checks = []
        all_d67_checks = []
        if "dominance_d50" in self.validators:
            all_d50_checks = self.validators["dominance_d50"].all_checks
        if "dominance_d67" in self.validators:
            all_d67_checks = self.validators["dominance_d67"].all_checks

        # Step 7: Build and return result
        return ValidationResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
            total_checks=len(self.validators),
            metadata={
                "description_metadata": metadata,
                "output_shape": output_df.shape,
                "supporting_shape": supporting_df.shape,
                "validators_run": list(self.validators.keys()),
                "all_d50_checks": all_d50_checks,
                "all_d67_checks": all_d67_checks,
                "thresholds": {
                    "d50": self.config["d50"],
                    "d67": self.config["d67"],
                    "min_n": self.config["min_n"],
                }
            }
        )

    def _parse_description(
        self,
        description: Union[str, DatasetMetadata, io.BytesIO],
        file_type: str
    ) -> DatasetMetadata:
        """Parse description document or return existing metadata"""
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
        file_type: str
    ) -> pd.DataFrame:
        """Load DataFrame from various sources"""
        if isinstance(data, pd.DataFrame):
            return data

        return self.data_loader.load_dataframe(data, file_type)

    def _extract_relevant_columns(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata
    ) -> pd.DataFrame:
        """
        Extract columns relevant for validation.

        If metadata specifies variables, use those.
        Otherwise, select all numeric columns.
        """
        if metadata.variables:
            # Use variables mentioned in metadata
            available_vars = [v for v in metadata.variables.keys() if v in df.columns]
            if available_vars:
                return df[available_vars]

        # Fallback: return all numeric columns
        return df.select_dtypes(include=[np.number])
