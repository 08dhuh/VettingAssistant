import streamlit as st
import pandas as pd
import os
from io import BytesIO
from ..checker_utils.regression_checker import RegressionChecker
from ..checker_utils.parsers.data_loader import DataLoader
from ..checker_utils.models import ViolationType, WarningType


def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment"""
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except:
        pass
    return os.environ.get('OPENAI_API_KEY')


def main():
    st.set_page_config(
        page_title="ABS Vetting Assistant - Regression Checker",
        page_icon=None,
        layout="wide"
    )

    st.title("ABS Vetting Assistant - Regression Checker")
    st.markdown("""
    This tool validates regression output datasets against statistical disclosure control rules:
    - **At least 10 degrees of freedom**: df = number of observations - number of independent variables 
    - **R-squared Check**: R² < 0.9 (or intercept must be suppressed)
    - **At least one numeric variable**: At least one numeric variable required (or intercept suppressed, or crosstab provided & checked for counts/dominance)
    """)

    st.caption("Single model analysis per output")
    # File uploads - three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Description")
        description_file = st.file_uploader(
            "Upload Data Description",
            type=['txt', 'pdf', 'docx', 'md'],
            help="Upload a file containing your data description, codebook, and variable definitions"
        )

    with col2:
        st.subheader("Output Data")
        output_file = st.file_uploader(
            "Upload Regression Output",
            type=['csv', 'xlsx'],
            help="Upload the regression output file containing coefficients, standard errors, N, and R²"
        )

    with col3:
        st.subheader("Supporting Data")
        supporting_file = st.file_uploader(
            "Upload Supporting Data",
            type=['csv', 'xlsx'],
            help="Required when all variables are categorical. Upload crosstab or supporting data."
        )

    # Preview uploaded files
    if output_file:
        with st.expander("Preview Output Data"):
            try:
                df = pd.read_csv(output_file) if output_file.name.endswith('.csv') else pd.read_excel(output_file)
                st.dataframe(df.head(10))
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                output_file.seek(0)  # Reset for later use
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if supporting_file:
        with st.expander("Preview Supporting Data"):
            try:
                df = pd.read_csv(supporting_file) if supporting_file.name.endswith('.csv') else pd.read_excel(supporting_file)
                st.dataframe(df.head(10))
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                supporting_file.seek(0)  # Reset for later use
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Run validation - supporting file is NOT required
    can_run = description_file is not None and output_file is not None
    if st.button("Run Validation Checks", type="primary", disabled=not can_run):
        if not can_run:
            st.error("Please upload description and output files before running validation.")
            return

        api_key = get_openai_api_key()
        if not api_key:
            st.warning("WARNING: OpenAI API key not found. Description parsing will be limited.")

        with st.spinner("Running validation checks..."):
            try:
                # Initialize checker
                checker = RegressionChecker(openai_api_key=api_key)

                # Get file types
                data_loader = DataLoader()
                desc_type = data_loader.get_file_type_from_name(description_file.name)
                output_type = data_loader.get_file_type_from_name(output_file.name)

                # Prepare supporting data if provided
                supporting_content = None
                support_type = None
                if supporting_file:
                    supporting_content = BytesIO(supporting_file.read())
                    support_type = data_loader.get_file_type_from_name(supporting_file.name)

                # Run validation
                result = checker.validate(
                    description=BytesIO(description_file.read()),
                    output=BytesIO(output_file.read()),
                    supporting=supporting_content,
                    description_file_type=desc_type,
                    output_file_type=output_type,
                    supporting_file_type=support_type,
                )

                # Display results
                st.markdown("---")
                st.header("Validation Results")

                # Overall status
                if result.passed:
                    st.success("**VALIDATION PASSED** - No violations found!")
                else:
                    st.error(f"**VALIDATION FAILED** - {len(result.violations)} violation(s) detected")

                st.info(f"**Summary:** {result.summary}")

                # Display warnings
                if result.has_warnings:
                    with st.expander(f"Warnings ({len(result.warnings)})", expanded=True):
                        for warning in result.warnings:
                            st.warning(f"**{warning.warning_type.value}**: {warning.message}")

                # Individual check results table
                st.markdown("### Check Results")

                if result.regression_metadata:
                    reg_meta = result.regression_metadata
                    thresholds = result.metadata.get("thresholds", {})

                    # Build check results
                    check_results = []

                    # Degrees of Freedom check
                    df_value = reg_meta.degrees_of_freedom
                    df_threshold = thresholds.get("min_df", 10)
                    df_passed = df_value >= df_threshold
                    df_violations = [v for v in result.violations if v.violation_type == ViolationType.INSUFFICIENT_DF]
                    check_results.append({
                        "Check": "Degrees of Freedom",
                        "Status": "PASS" if df_passed else "FAIL",
                        "Details": f"df = {df_value:,} {'≥' if df_passed else '<'} {df_threshold}",
                        "_passed": df_passed
                    })

                    # R-squared check (for OLS)
                    if reg_meta.regression_type == "OLS":
                        r2_value = reg_meta.r_squared
                        r2_threshold = 0.9
                        r2_passed = r2_value < r2_threshold or not reg_meta.has_intercept
                        r2_details = f"R² = {r2_value:.4f}"
                        if r2_value >= r2_threshold:
                            r2_details += f" ≥ {r2_threshold}"
                            if reg_meta.has_intercept:
                                r2_details += ", intercept present"
                            else:
                                r2_details += ", intercept suppressed (OK)"
                        else:
                            r2_details += f" < {r2_threshold}"
                        check_results.append({
                            "Check": "R-squared",
                            "Status": "PASS" if r2_passed else "FAIL",
                            "Details": r2_details,
                            "_passed": r2_passed
                        })

                    # Numeric Variable check
                    intercept_names = {"constant", "intercept", "_cons", "(intercept)"}
                    independent_vars = [c for c in reg_meta.coefficients if c.variable_name.lower() not in intercept_names]
                    continuous_vars = [c for c in independent_vars if c.variable_type == "continuous"]

                    # Check for violations: NUMERIC_VARIABLE (no supporting data) or RULE_OF_10
                    nv_violations = [v for v in result.violations if v.violation_type == ViolationType.NUMERIC_VARIABLE]
                    rule_of_10_violations = [v for v in result.violations if v.violation_type == ViolationType.RULE_OF_10]
                    nv_passed = len(nv_violations) == 0 and len(rule_of_10_violations) == 0

                    if continuous_vars:
                        nv_details = f"{', '.join(c.variable_name for c in continuous_vars)} are continuous"
                    else:
                        if nv_violations:
                            nv_details = "No continuous variables - supporting data required"
                        elif rule_of_10_violations:
                            failed_vars = [v.context.get("variable", v.column) for v in rule_of_10_violations]
                            nv_details = f"Rule of 10 failed for: {', '.join(failed_vars)}"
                        else:
                            nv_details = "All categorical variables pass Rule of 10"

                    check_results.append({
                        "Check": "Numeric Variable",
                        "Status": "PASS" if nv_passed else "FAIL",
                        "Details": nv_details,
                        "_passed": nv_passed
                    })

                    # Display check results table
                    checks_df = pd.DataFrame(check_results)

                    def style_check_results(row):
                        passed = checks_df.loc[row.name, '_passed']
                        if passed:
                            return ['background-color: #d4edda; color: #155724' for _ in row]
                        else:
                            return ['background-color: #f8d7da; color: #721c24' for _ in row]

                    display_df = checks_df.drop(columns=['_passed'])
                    styled_df = display_df.style.apply(style_check_results, axis=1)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

                    # Parsed Regression Output table
                    st.markdown("### Parsed Regression Output")

                    # Intercept names to check
                    intercept_names = {"constant", "intercept", "_cons", "(intercept)"}

                    coef_data = []
                    for coef in reg_meta.coefficients:
                        stars = "*" * coef.significance_level
                        # Intercept is not a variable - leave Type blank
                        is_intercept = coef.variable_name.lower() in intercept_names
                        if is_intercept:
                            var_type = "-"
                        elif coef.variable_type:
                            var_type = coef.variable_type
                        else:
                            var_type = "unknown"

                        coef_data.append({
                            "Variable": coef.variable_name,
                            "Coefficient": f"{coef.coefficient:.6f}",
                            "Std. Error": f"{coef.std_error:.6f}" if coef.std_error else "-",
                            "Significance": stars if stars else "-",
                            "Type": var_type
                        })

                    coef_df = pd.DataFrame(coef_data)

                    # Style the type column
                    def style_variable_type(val):
                        if val == "continuous":
                            return "background-color: #cce5ff; color: #004085"
                        elif val == "binary":
                            return "background-color: #fff3cd; color: #856404"
                        elif val == "unknown":
                            return "background-color: #e9ecef; color: #495057"
                        return ""

                    styled_coef_df = coef_df.style.map(style_variable_type, subset=['Type'])
                    st.dataframe(styled_coef_df, use_container_width=True, hide_index=True)

                    # Display Rule of 10 summary if applicable (all categorical variables)
                    if not continuous_vars and supporting_file:
                        st.markdown("### Rule of 10 Check (Supporting Data)")

                        rule_of_10_results = result.metadata.get("rule_of_10_results", [])
                        if rule_of_10_results:
                            r10_summary = []
                            for check in rule_of_10_results:
                                is_fail = check["status"] == "fail"
                                r10_summary.append({
                                    "Variable": check["variable"],
                                    "Count": f"{check['count']:,}" if check["count"] is not None else "-",
                                    "Threshold": "≥ 10",
                                    "Status": "FAIL" if is_fail else ("PASS" if check["status"] == "pass" else "NOT FOUND"),
                                    "_is_violation": is_fail or check["status"] == "not_found"
                                })

                            if r10_summary:
                                r10_df = pd.DataFrame(r10_summary)

                                def style_rule_of_10(row):
                                    is_fail = r10_df.loc[row.name, '_is_violation'] if '_is_violation' in r10_df.columns else False
                                    if is_fail:
                                        return ['background-color: #f8d7da; color: #721c24' for _ in row]
                                    return ['background-color: #d4edda; color: #155724' for _ in row]

                                display_r10_df = r10_df.drop(columns=['_is_violation'])
                                styled_r10_df = display_r10_df.style.apply(style_rule_of_10, axis=1)
                                st.dataframe(styled_r10_df, use_container_width=True, hide_index=True)

                                st.markdown("""
                                **Rule of 10**: Each categorical variable's count must be ≥ 10 to avoid disclosure risk.
                                - Rows with **red background** indicate violations or missing data
                                """)

                # Display violations by type
                if result.violations:
                    st.markdown("### Detailed Violations")

                    # Group violations by type
                    violations_by_type = {}
                    for v in result.violations:
                        if v.violation_type not in violations_by_type:
                            violations_by_type[v.violation_type] = []
                        violations_by_type[v.violation_type].append(v)

                    # Display each type
                    for vtype, violations in violations_by_type.items():
                        with st.expander(f"{vtype.value.upper()} - {len(violations)} violation(s)", expanded=True):
                            for v in violations:
                                st.error(f"**{vtype.value}**: {v.context.get('message', str(v))}")

                                # Show additional context
                                context_items = {k: val for k, val in v.context.items() if k != 'message'}
                                if context_items:
                                    st.json(context_items)

                # Display metadata
                with st.expander("Metadata & Configuration"):
                    st.json({
                        "n_observations": result.metadata.get("n_observations"),
                        "r_squared": result.metadata.get("r_squared"),
                        "degrees_of_freedom": result.metadata.get("degrees_of_freedom"),
                        "n_independent_vars": result.metadata.get("n_independent_vars"),
                        "has_intercept": result.metadata.get("has_intercept"),
                        "has_continuous_variable": result.metadata.get("has_continuous_variable"),
                        "thresholds": result.metadata.get("thresholds"),
                    })

                    # Show parsed description metadata if available
                    if result.description_metadata and api_key:
                        st.markdown("**Parsed Description Metadata:**")
                        desc_meta = result.description_metadata
                        st.json({
                            "Population": desc_meta.population,
                            "Method": desc_meta.method_of_analysis,
                            "Datasets": desc_meta.datasets_used,
                            "Variables": desc_meta.variables,
                            "Variable Types": desc_meta.variable_types,
                            "Regression Type": desc_meta.regression_type,
                            "Outcome Variable": desc_meta.outcome_variable,
                        })

            except Exception as e:
                st.error(f"Error during validation: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
