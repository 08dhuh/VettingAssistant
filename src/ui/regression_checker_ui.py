import streamlit as st
import pandas as pd
import os
from io import BytesIO
from ..checker_utils.regression_checker import RegressionChecker
from ..checker_utils.parsers.data_loader import DataLoader
from ..checker_utils.models import ViolationType, WarningType


# Maximum number of models allowed
MAX_MODELS = 5


def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment"""
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except:
        pass
    return os.environ.get('OPENAI_API_KEY')


def render_model_row(model_index: int):
    """Render a single model upload row with 3 columns."""
    st.markdown(f"**Model {model_index + 1}**")
    col1, col2, col3 = st.columns(3)

    with col1:
        description_file = st.file_uploader(
            "Description",
            type=['txt', 'pdf', 'docx', 'md'],
            help="Upload a file containing your data description, codebook, and variable definitions",
            key=f"desc_{model_index}"
        )

    with col2:
        output_file = st.file_uploader(
            "Output",
            type=['csv', 'xlsx'],
            help="Upload the regression output file containing coefficients, standard errors, N, and R²",
            key=f"output_{model_index}"
        )

    with col3:
        supporting_file = st.file_uploader(
            "Supporting (Optional)",
            type=['csv', 'xlsx'],
            help="Required when all variables are categorical. Upload crosstab or supporting data.",
            key=f"support_{model_index}"
        )

    return {
        "description": description_file,
        "output": output_file,
        "supporting": supporting_file,
    }


def display_single_model_result(result, model_id: str, supporting_file_provided: bool):
    """Display validation results for a single model."""
    st.markdown(f"#### {model_id}")

    # Overall status
    if result.passed:
        st.success(f"**PASSED** - No violations found!")
    else:
        st.error(f"**FAILED** - {len(result.violations)} violation(s) detected")

    # Display warnings
    if result.has_warnings:
        with st.expander(f"Warnings ({len(result.warnings)})", expanded=False):
            for warning in result.warnings:
                st.warning(f"**{warning.warning_type.value}**: {warning.message}")

    # Check results table
    if result.regression_metadata:
        reg_meta = result.regression_metadata
        thresholds = result.metadata.get("thresholds", {})

        # Build check results
        check_results = []

        # Degrees of Freedom check
        df_value = reg_meta.degrees_of_freedom
        df_threshold = thresholds.get("min_df", 10)
        df_passed = df_value >= df_threshold
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

        # Parsed Regression Output table in expander
        with st.expander("Parsed Regression Output"):
            coef_data = []
            for coef in reg_meta.coefficients:
                stars = "*" * coef.significance_level
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

        # Display Rule of 10 summary if applicable
        if not continuous_vars and supporting_file_provided:
            rule_of_10_results = result.metadata.get("rule_of_10_results", [])
            if rule_of_10_results:
                with st.expander("Rule of 10 Check (Supporting Data)"):
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

    # Display violations by type
    if result.violations:
        with st.expander(f"Detailed Violations ({len(result.violations)})", expanded=True):
            for v in result.violations:
                st.error(f"**{v.violation_type.value}**: {v.context.get('message', str(v))}")


def display_comparison_results(comparison_results):
    """Display multi-model comparison results."""
    st.markdown("## Multi-Model Comparison")
    st.markdown("### Observation Count Differences")

    # Create comparison table
    comparison_data = []
    for result in comparison_results:
        n_a_str = f"{result.n_a:,}" if result.n_a is not None else "Suppressed"
        n_b_str = f"{result.n_b:,}" if result.n_b is not None else "Suppressed"
        diff_str = f"{result.difference:,}" if result.difference is not None else "—"

        if result.is_subset is True:
            subset_str = "Yes"
        elif result.is_subset is False:
            subset_str = "No"
        else:
            subset_str = "—"

        if result.passed is True:
            status_str = "✓ PASS"
        elif result.passed is False:
            status_str = "✗ FAIL"
        else:
            status_str = "⚠️ CHECK"

        comparison_data.append({
            "Models": f"{result.model_a_id} vs {result.model_b_id}",
            "N (A)": n_a_str,
            "N (B)": n_b_str,
            "Difference": diff_str,
            "Subset?": subset_str,
            "Status": status_str,
            "_passed": result.passed
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Style failed rows
    def style_comparison_row(row):
        passed = comparison_df.loc[row.name, '_passed']
        if passed is False:
            return ['background-color: #f8d7da; color: #721c24' for _ in row]
        elif passed is None:
            return ['background-color: #fff3cd; color: #856404' for _ in row]
        return ['background-color: #d4edda; color: #155724' for _ in row]

    display_comparison_df = comparison_df.drop(columns=['_passed'])
    styled_comparison_df = display_comparison_df.style.apply(style_comparison_row, axis=1)
    st.dataframe(styled_comparison_df, use_container_width=True, hide_index=True)

    # Show violations prominently
    failures = [r for r in comparison_results if r.passed is False]
    if failures:
        st.error(f"⚠️ {len(failures)} observation count violation(s) detected")
        for f in failures:
            st.warning(f"**{f.model_a_id} vs {f.model_b_id}**: {f.message}")

    # Show checks that couldn't be performed
    checks_needed = [r for r in comparison_results if r.passed is None]
    if checks_needed:
        for c in checks_needed:
            st.info(f"**{c.model_a_id} vs {c.model_b_id}**: {c.message}")

    st.markdown("""
    **Observation Difference Rule**: If observation counts differ by < 10 between models
    where one model's variables are a subset of the other, either suppress counts or justify that models are not subsets of each other.
    """)


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
    - **Multi-model comparison**: Observation count differences < 10 between subset models are flagged
    """)

    # Initialize session state for model count
    if 'model_count' not in st.session_state:
        st.session_state.model_count = 1

    # Model management header
    col_header, col_add = st.columns([4, 1])
    with col_header:
        st.subheader("Regression Models")
    with col_add:
        if st.session_state.model_count < MAX_MODELS:
            if st.button("+ Add Model", use_container_width=True):
                st.session_state.model_count += 1
                st.rerun()
        else:
            st.button("+ Add Model", disabled=True, use_container_width=True, help=f"Maximum {MAX_MODELS} models")

    # Render model upload rows
    uploaded_models = []
    for i in range(st.session_state.model_count):
        model_files = render_model_row(i)
        uploaded_models.append(model_files)
        if i < st.session_state.model_count - 1:
            st.divider()

    # Remove last model button (only show if > 1 model)
    if st.session_state.model_count > 1:
        if st.button("− Remove Last Model"):
            st.session_state.model_count -= 1
            st.rerun()

    # Count valid models (those with output file)
    valid_models = [m for m in uploaded_models if m["output"] is not None]
    valid_count = len(valid_models)

    # Show model count info
    st.markdown("---")
    if valid_count == 0:
        st.info("ℹ️ Upload at least one output file to run validation.")
    elif valid_count == 1:
        st.info(f"ℹ️ {valid_count} model ready for validation. Single model analysis.")
    else:
        st.info(f"ℹ️ {valid_count} model(s) ready for validation. Multi-model comparison will be performed.")

    # Run validation button
    can_run = valid_count >= 1
    if st.button("Run Validation Checks", type="primary", disabled=not can_run):
        api_key = get_openai_api_key()
        if not api_key:
            st.warning("WARNING: OpenAI API key not found. Description parsing will be limited.")

        with st.spinner("Running validation checks..."):
            try:
                # Initialize checker
                checker = RegressionChecker(openai_api_key=api_key)
                data_loader = DataLoader()

                # Prepare models for validation
                models_to_validate = []
                supporting_file_flags = []  # Track which models had supporting files

                for i, model_files in enumerate(uploaded_models):
                    if model_files["output"] is None:
                        continue

                    model_id = f"Model {i + 1}"

                    # Get file types
                    output_type = data_loader.get_file_type_from_name(model_files["output"].name)

                    # Prepare description
                    description_content = None
                    desc_type = "txt"
                    if model_files["description"]:
                        description_content = BytesIO(model_files["description"].read())
                        desc_type = data_loader.get_file_type_from_name(model_files["description"].name)

                    # Prepare supporting data
                    supporting_content = None
                    support_type = None
                    if model_files["supporting"]:
                        supporting_content = BytesIO(model_files["supporting"].read())
                        support_type = data_loader.get_file_type_from_name(model_files["supporting"].name)

                    models_to_validate.append({
                        "model_id": model_id,
                        "description": description_content,
                        "output": BytesIO(model_files["output"].read()),
                        "supporting": supporting_content,
                        "description_file_type": desc_type,
                        "output_file_type": output_type,
                        "supporting_file_type": support_type,
                    })
                    supporting_file_flags.append(model_files["supporting"] is not None)

                # Run validation
                if len(models_to_validate) == 1:
                    # Single model validation
                    model_data = models_to_validate[0]
                    result = checker.validate(
                        description=model_data["description"],
                        output=model_data["output"],
                        supporting=model_data["supporting"],
                        description_file_type=model_data["description_file_type"],
                        output_file_type=model_data["output_file_type"],
                        supporting_file_type=model_data["supporting_file_type"],
                    )
                    result.model_id = model_data["model_id"]

                    # Display results
                    st.markdown("---")
                    st.header("Validation Results")

                    # Overall status
                    if result.passed:
                        st.success("**VALIDATION PASSED** - No violations found!")
                    else:
                        st.error(f"**VALIDATION FAILED** - {len(result.violations)} violation(s) detected")

                    st.info(f"**Summary:** {result.summary}")

                    display_single_model_result(result, model_data["model_id"], supporting_file_flags[0])

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

                else:
                    # Multi-model validation
                    multi_result = checker.validate_multiple(models_to_validate)

                    # Display results
                    st.markdown("---")
                    st.header("Validation Results")

                    # Overall status
                    if multi_result.overall_passed:
                        st.success("**VALIDATION PASSED** - All models passed and no comparison violations!")
                    else:
                        individual_failed = sum(1 for r in multi_result.individual_results if not r.passed)
                        comparison_failed = sum(1 for r in multi_result.comparison_results if r.passed is False)
                        st.error(
                            f"**VALIDATION FAILED** - {individual_failed} model(s) failed, "
                            f"{comparison_failed} comparison violation(s)"
                        )

                    st.info(f"**Summary:** {multi_result.summary}")

                    # Display individual model results
                    st.markdown("## Individual Model Results")
                    for i, result in enumerate(multi_result.individual_results):
                        display_single_model_result(
                            result,
                            result.model_id or f"Model {i + 1}",
                            supporting_file_flags[i]
                        )
                        if i < len(multi_result.individual_results) - 1:
                            st.divider()

                    # Display comparison results
                    if multi_result.comparison_results:
                        display_comparison_results(multi_result.comparison_results)

            except Exception as e:
                st.error(f"Error during validation: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
