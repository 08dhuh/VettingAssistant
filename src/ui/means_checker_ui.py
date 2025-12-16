import streamlit as st
import pandas as pd
import os
from io import BytesIO
from ..checker_utils.means_checker import MeansChecker
from ..checker_utils.parsers.data_loader import DataLoader
from ..checker_utils.models import ViolationType


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
        page_title="ABS Vetting Assistant - Means Checker",
        page_icon=None,
        layout="wide"
    )

    st.title("ABS Vetting Assistant - Means Checker")
    st.markdown("""
    This tool validates means/averages datasets against statistical disclosure control rules:
    - **Rule of N**: Ensures sample sizes meet minimum threshold (default: 10)
    - **Dominance D50**: Checks if the highest value exceeds 50% of total
    - **Dominance D67**: Checks if top 2 values exceed 67% of total
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        min_n = st.number_input(
            "Minimum N Threshold",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum sample size required (Rule of N)",
            disabled=True
        )
        d50_thresh = st.slider(
            "D50 Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="Maximum ratio for single highest value",
            disabled=True
            
        )
        d67_thresh = st.slider(
            "D67 Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.67,
            step=0.05,
            help="Maximum ratio for sum of top 2 values",
            disabled=True
            
        )

    # File uploads
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
            "Upload Output (Means)",
            type=['csv', 'xlsx'],
            help="Upload the output file containing calculated means/averages"
        )

    with col3:
        st.subheader("Supporting Data")
        supporting_file = st.file_uploader(
            "Upload Supporting Data",
            type=['csv', 'xlsx'],
            help="Upload supporting data with counts, individual values, or first/second highest values"
        )

    # Preview uploaded files
    if output_file:
        with st.expander("Preview Output Data"):
            df = pd.read_csv(output_file) if output_file.name.endswith('.csv') else pd.read_excel(output_file)
            st.dataframe(df.head(10))
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            output_file.seek(0)  # Reset for later use

    if supporting_file:
        with st.expander("Preview Supporting Data"):
            df = pd.read_csv(supporting_file) if supporting_file.name.endswith('.csv') else pd.read_excel(supporting_file)
            st.dataframe(df.head(10))
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            supporting_file.seek(0)  # Reset for later use

    # Run validation
    if st.button("Run Validation Checks", type="primary", disabled=not all([description_file, output_file, supporting_file])):
        if not all([description_file, output_file, supporting_file]):
            st.error("Please upload all three files before running validation.")
            return

        api_key = get_openai_api_key()
        if not api_key:
            st.warning("WARNING: OpenAI API key not found. Description parsing will be limited.")

        with st.spinner("Running validation checks..."):
            try:
                # Initialize checker
                checker = MeansChecker(
                    min_n_threshold=min_n,
                    d50_threshold=d50_thresh,
                    d67_threshold=d67_thresh,
                    openai_api_key=api_key
                )

                # Get file types
                data_loader = DataLoader()
                desc_type = data_loader.get_file_type_from_name(description_file.name)
                output_type = data_loader.get_file_type_from_name(output_file.name)
                support_type = data_loader.get_file_type_from_name(supporting_file.name)

                # Run validation
                result = checker.validate(
                    description=BytesIO(description_file.read()),
                    output=BytesIO(output_file.read()),
                    supporting=BytesIO(supporting_file.read()),
                    description_file_type=desc_type,
                    output_file_type=output_type,
                    supporting_file_type=support_type,
                )

                # Display results
                st.markdown("---")
                st.header("Validation Results")

                # Overall status
                if result.passed:
                    st.success(f"**VALIDATION PASSED** - No violations found!")
                else:
                    st.error(f"**VALIDATION FAILED** - {len(result.violations)} violation(s) detected")

                st.info(f"**Summary:** {result.summary}")

                # Display warnings
                if result.has_warnings:
                    with st.expander(f"Warnings ({len(result.warnings)})", expanded=True):
                        for warning in result.warnings:
                            st.warning(f"**{warning.warning_type.value}**: {warning.message}")

                # Display dominance calculation summary
                st.markdown("### Dominance Check Summary")

                # Extract all dominance checks from metadata
                all_d50_checks = result.metadata.get("all_d50_checks", [])
                all_d67_checks = result.metadata.get("all_d67_checks", [])

                # Get thresholds from metadata
                d50_threshold = result.metadata.get("thresholds", {}).get("d50", 0.50)
                d67_threshold = result.metadata.get("thresholds", {}).get("d67", 0.67)

                # Prepare D50 summary
                d50_summary = []
                for check in all_d50_checks:
                    # Check threshold directly from the ratio
                    is_violation = check['d50_ratio'] > d50_threshold
                    d50_summary.append({
                        "Row": check['row'],
                        "Mean": f"${check['mean']:,.2f}",
                        "Observations": f"{check['observations']:,.0f}",
                        "Total Sum": f"${check['total_sum']:,.2f}",
                        "Highest Value": f"${check['highest_value']:,.2f}",
                        "D50 Ratio": f"{check['d50_ratio']*100:.2f}%",
                        "Status": "FAIL" if is_violation else "PASS",
                        "_is_violation": is_violation
                    })

                # Prepare D67 summary
                d67_summary = []
                for check in all_d67_checks:
                    # Check threshold directly from the ratio
                    is_violation = check['d67_ratio'] > d67_threshold
                    d67_summary.append({
                        "Row": check['row'],
                        "Mean": f"${check['mean']:,.2f}",
                        "Observations": f"{ check['observations']:,.0f}",
                        "Total Sum": f"${check['total_sum']:,.2f}",
                        "First + Second": f"${check['d67_sum']:,.2f}",
                        "D67 Ratio": f"{check['d67_ratio']*100:.2f}%",
                        "Status": "FAIL" if is_violation else "PASS",
                        "_is_violation": is_violation
                    })

                # Display D50 checks
                if d50_summary:
                    st.markdown("#### D50 Dominance Checks (Threshold: ≤ 50%)")
                    d50_df = pd.DataFrame(d50_summary)

                    # Create styled dataframe
                    # Keep the _is_violation for styling, then drop it
                    def style_violations(row):
                        is_fail = d50_df.loc[row.name, '_is_violation'] if '_is_violation' in d50_df.columns else False
                        return ['background-color: #ffcccc' if is_fail else '' for _ in row]

                    display_df = d50_df.drop(columns=['_is_violation'])
                    styled_df = display_df.style.apply(style_violations, axis=1)
                    st.dataframe(styled_df, use_container_width=True)

                # Display D67 checks
                if d67_summary:
                    st.markdown("#### D67 Dominance Checks (Threshold: ≤ 67%)")
                    d67_df = pd.DataFrame(d67_summary)

                    # Create styled dataframe
                    def style_violations_d67(row):
                        is_fail = d67_df.loc[row.name, '_is_violation'] if '_is_violation' in d67_df.columns else False
                        return ['background-color: #ffcccc' if is_fail else '' for _ in row]

                    display_df = d67_df.drop(columns=['_is_violation'])
                    styled_df = display_df.style.apply(style_violations_d67, axis=1)
                    st.dataframe(styled_df, use_container_width=True)

                if d50_summary or d67_summary:
                    st.markdown("""
                    **Calculation Details:**
                    - **Total Sum** = Mean × Observations
                    - **D50 Ratio** = Highest Value / Total Sum (should be ≤ 50%)
                    - **D67 Ratio** = (First + Second Highest) / Total Sum (should be ≤ 67%)
                    - Rows with **red background** indicate violations
                    """)
                else:
                    st.info("No dominance checks were performed.")

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
                            # Convert to DataFrame for display
                            violations_data = []
                            for v in violations:
                                violations_data.append({
                                    "Row": v.row,
                                    "Column": v.column,
                                    "Value": v.value,
                                    **{k: v for k, v in v.context.items() if k not in ['position']}
                                })

                            violations_df = pd.DataFrame(violations_data)
                            st.dataframe(violations_df, use_container_width=True)

                            # Summary stats
                            if not violations_df.empty and 'Column' in violations_df.columns:
                                st.markdown("**Violations by Column:**")
                                col_counts = violations_df['Column'].value_counts()
                                for col, count in col_counts.items():
                                    st.write(f"- {col}: {count}")

                # Display metadata
                with st.expander("Metadata & Configuration"):
                    st.json({
                        "Validators Run": result.metadata.get("validators_run", []),
                        "Total Checks": result.total_checks,
                        "Output Shape": result.metadata.get("output_shape"),
                        "Supporting Shape": result.metadata.get("supporting_shape"),
                        "Thresholds": {
                            "Rule of N": min_n,
                            "D50": d50_thresh,
                            "D67": d67_thresh,
                        }
                    })

                    # Show parsed metadata if available
                    desc_metadata = result.metadata.get("description_metadata")
                    if desc_metadata and api_key:
                        st.markdown("**Parsed Description Metadata:**")
                        st.json({
                            "Population": desc_metadata.population,
                            "Method": desc_metadata.method_of_analysis,
                            "Datasets": desc_metadata.datasets_used,
                            "Variables": desc_metadata.variables,
                            "Percentage Columns": desc_metadata.percentage_columns,
                            "Sample Size Column": desc_metadata.sample_size_column,
                        })

            except Exception as e:
                st.error(f"Error during validation: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()