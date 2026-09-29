"""
File Data Analysis System
==========================
College Mini-Project in Data Analysis and Essentials
Domain: Data Analytics / File-Based Data Analysis

Technology Stack:
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

from modules.file_handler import (
    load_dataset,
    get_dataset_metadata,
    get_preview
)
from modules.processing import (
    classify_columns,
    get_column_summary_table,
    format_memory_size,
    safe_convert_column
)
from modules.cleaning import (
    get_data_quality_report,
    remove_duplicates,
    handle_missing_numerical,
    handle_missing_categorical,
    trim_whitespace,
    compare_before_after
)
from modules.exploration import (
    search_dataset,
    filter_dataset,
    sort_dataset,
    paginate_dataframe
)
from modules.statistics import (
    get_single_column_stats,
    get_all_numerical_summary,
    get_categorical_summary,
    get_correlation_matrix
)
from modules.visualization import (
    plot_bar_chart,
    plot_histogram,
    plot_boxplot,
    plot_scatterplot,
    plot_line_chart,
    plot_correlation_heatmap
)
from modules.insights import generate_factual_insights
from modules.reporting import generate_text_report, get_csv_download_bytes


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="File Data Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional student project presentation
st.markdown("""
<style>
    /* Global accents */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e40af !important;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #334155 !important;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563eb !important;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #334155 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .workflow-box {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: #f8fafc !important;
    }
    .workflow-title {
        color: #60a5fa !important;
        font-size: 1.1rem;
        font-weight: 700;
        display: block;
        margin-bottom: 10px;
    }
    .workflow-step {
        display: inline-block;
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 700;
        font-family: monospace;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 6px;
    }
    .workflow-step-final {
        display: inline-block;
        background-color: #10b981 !important;
        color: #ffffff !important;
        font-weight: 700;
        font-family: monospace;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 6px;
    }
    .workflow-arrow {
        color: #94a3b8 !important;
        font-weight: 900;
        font-size: 1.1rem;
        margin: 0 4px;
    }
    .insight-card {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .insight-card strong {
        color: #0f172a !important;
    }
    .team-badge {
        font-size: 0.75rem;
        background: #e0e7ff !important;
        color: #3730a3 !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = None
if 'current_df' not in st.session_state:
    st.session_state['current_df'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = "No dataset loaded"
if 'cleaning_logs' not in st.session_state:
    st.session_state['cleaning_logs'] = []
if 'saved_charts' not in st.session_state:
    st.session_state['saved_charts'] = []


def load_selected_dataframe(df: pd.DataFrame, name: str):
    """Safely store dataset into session state, preserving the original copy."""
    st.session_state['original_df'] = df.copy()
    st.session_state['current_df'] = df.copy()
    st.session_state['file_name'] = name
    st.session_state['cleaning_logs'] = []
    st.session_state['saved_charts'] = []


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & PRESENTATION GUIDE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📁 Analysis System")
    st.caption("College Mini Project")

    # Navigation Menu
    pages = [
        "Home",
        "Dataset",
        "Cleaning",
        "Explore",
        "Statistics",
        "Visualization",
        "Report"
    ]
    page = st.radio("Navigation", pages, index=0)

    st.markdown("---")

    # Current Dataset Status Widget
    if st.session_state['current_df'] is not None:
        st.markdown("**Active Dataset:**")
        st.success(f"📄 {st.session_state['file_name']}")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", f"{len(st.session_state['current_df']):,}")
        with c2:
            st.metric("Cols", len(st.session_state['current_df'].columns))
    else:
        st.info("No dataset loaded yet.")

    st.markdown("---")

    # Four-Person Presentation Viva Guide
    with st.expander("👥 Presentation Roles Guide", expanded=False):
        st.markdown("""
        **Person 1 (Team Leader):**
        * Pages: Home → Dataset
        * Objective 1: File Automation

        **Person 2:**
        * Page: Cleaning
        * Objective 2: Data Quality

        **Person 3:**
        * Pages: Explore → Statistics
        * Objective 3: Search, Filter, Stats

        **Person 4:**
        * Pages: Visualization → Report
        * Objective 4: Insights & Reporting
        """)


# =============================================================================
# 1. HOME PAGE
# =============================================================================
if page == "Home":
    st.markdown('<div class="main-title">File Data Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transform raw file-based data into meaningful insights.</div>', unsafe_allow_html=True)

    st.markdown("""
    > **Upload a tabular dataset, clean it, explore it, analyze it, and visualize meaningful information.**
    """)

    # Visual Workflow
    st.markdown("""
    <div class="workflow-box" style="background-color: #1e293b !important; color: #f8fafc !important; border: 1px solid #334155 !important; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
        <span class="workflow-title" style="color: #60a5fa !important; font-size: 1.1rem; font-weight: 700; display: block; margin-bottom: 12px;">Application Workflow Pipeline:</span>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px;">
            <span class="workflow-step" style="background-color: #2563eb !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">UPLOAD</span>
            <span class="workflow-arrow" style="color: #94a3b8 !important; font-weight: 900; font-size: 1.1rem;">➔</span>
            <span class="workflow-step" style="background-color: #2563eb !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">PROCESS</span>
            <span class="workflow-arrow" style="color: #94a3b8 !important; font-weight: 900; font-size: 1.1rem;">➔</span>
            <span class="workflow-step" style="background-color: #2563eb !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">CLEAN</span>
            <span class="workflow-arrow" style="color: #94a3b8 !important; font-weight: 900; font-size: 1.1rem;">➔</span>
            <span class="workflow-step" style="background-color: #2563eb !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">EXPLORE</span>
            <span class="workflow-arrow" style="color: #94a3b8 !important; font-weight: 900; font-size: 1.1rem;">➔</span>
            <span class="workflow-step" style="background-color: #2563eb !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">ANALYZE</span>
            <span class="workflow-arrow" style="color: #94a3b8 !important; font-weight: 900; font-size: 1.1rem;">➔</span>
            <span class="workflow-step" style="background-color: #2563eb !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">VISUALIZE</span>
            <span class="workflow-arrow" style="color: #94a3b8 !important; font-weight: 900; font-size: 1.1rem;">➔</span>
            <span class="workflow-step-final" style="background-color: #10b981 !important; color: #ffffff !important; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-family: monospace;">INSIGHTS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Objectives
    st.subheader("Project Objectives")
    ob1, ob2 = st.columns(2)
    with ob1:
        st.info("**Objective 1 — Automation:** To automate data processing and analysis from CSV and text files, reducing the time and effort required for manual examination.")
        st.info("**Objective 2 — Cleaning:** To clean and improve data quality by identifying and handling missing values, duplicate records, and other data inconsistencies.")
    with ob2:
        st.info("**Objective 3 — Exploration & Stats:** To provide efficient data exploration through searching, filtering, sorting, and statistical analysis such as mean, median, minimum, and maximum.")
        st.info("**Objective 4 — Visualization & Insights:** To present meaningful insights clearly using summaries, tables, and visualizations to help users understand patterns, trends, and important information.")

    st.markdown("---")

    # Data Input Section
    col_upload, col_sample = st.columns([1.2, 1])

    with col_upload:
        st.subheader("📤 Upload Dataset")
        st.caption("Supports CSV and tabular text files (.csv, .txt, .tsv)")
        uploaded_file = st.file_uploader("Select tabular file to analyze", type=["csv", "txt", "tsv"])

        if uploaded_file is not None:
            df, error = load_dataset(uploaded_file, uploaded_file.name)
            if error:
                st.error(error)
            else:
                load_selected_dataframe(df, uploaded_file.name)
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df):,} rows, {len(df.columns)} columns)!")
                st.info("👉 Switch to the **Dataset** tab in the sidebar to inspect the data.")

    with col_sample:
        st.subheader("🚀 Try Example Datasets")
        st.caption("Click to load pre-configured demo datasets across diverse domains:")

        # Primary Example
        if st.button("🎓 Example Dataset — Student Performance", use_container_width=True, type="primary"):
            sample_path = os.path.join("data", "student_performance.csv")
            df, error = load_dataset(sample_path, "student_performance.csv")
            if not error and df is not None:
                load_selected_dataframe(df, "student_performance.csv (Demo)")
                st.success("Loaded Example Dataset: Student Performance (Education Domain)")
            else:
                st.error(f"Error loading demo: {error}")

        # Secondary Domain 1: Sales
        if st.button("💼 Example Dataset — Sales & Orders", use_container_width=True):
            sample_path = os.path.join("data", "sales_data.csv")
            df, error = load_dataset(sample_path, "sales_data.csv")
            if not error and df is not None:
                load_selected_dataframe(df, "sales_data.csv (Demo)")
                st.success("Loaded Example Dataset: Sales Data (Commercial Domain)")

        # Secondary Domain 2: Employee HR
        if st.button("🏢 Example Dataset — Employee HR", use_container_width=True):
            sample_path = os.path.join("data", "employee_data.csv")
            df, error = load_dataset(sample_path, "employee_data.csv")
            if not error and df is not None:
                load_selected_dataframe(df, "employee_data.csv (Demo)")
                st.success("Loaded Example Dataset: Employee Data (HR Domain)")

        # Secondary Domain 3: Tabular Text (Tab delimited)
        if st.button("📑 Example Dataset — Customer Data (Tab Delimited TXT)", use_container_width=True):
            sample_path = os.path.join("data", "customer_data.txt")
            df, error = load_dataset(sample_path, "customer_data.txt")
            if not error and df is not None:
                load_selected_dataframe(df, "customer_data.txt (Demo)")
                st.success("Loaded Example Dataset: Customer Data (Tab-delimited Text)")

        # Review 2 College Notebook Datasets
        if st.button("📘 Review 2 Dataset — Student Performance (Clean)", use_container_width=True):
            sample_path = os.path.join("data", "student_performance_review2.csv")
            df, error = load_dataset(sample_path, "student_performance_review2.csv")
            if not error and df is not None:
                load_selected_dataframe(df, "student_performance_review2.csv (Review 2)")
                st.success("Loaded Review 2 Student Performance Dataset (Clean)")
            else:
                st.error(f"Error loading demo: {error}")

        if st.button("🧪 Review 2 Dataset — Raw with Anomalies (For Cleaning Demo)", use_container_width=True):
            sample_path = os.path.join("data", "student_performance_review2_uncleaned.csv")
            df, error = load_dataset(sample_path, "student_performance_review2_uncleaned.csv")
            if not error and df is not None:
                load_selected_dataframe(df, "student_performance_review2_uncleaned.csv (Review 2 Raw)")
                st.success("Loaded Review 2 Student Performance with Missing Values & Duplicate Row!")
            else:
                st.error(f"Error loading demo: {error}")


# =============================================================================
# 2. DATASET PAGE (Person 1 — Team Leader)
# =============================================================================
elif page == "Dataset":
    st.markdown('<div class="team-badge">PERSON 1 — TEAM LEADER | OBJECTIVE 1: AUTOMATION</div>', unsafe_allow_html=True)
    st.title("Dataset Overview & Structure Inspection")

    if st.session_state['current_df'] is None:
        st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload or select an example dataset.")
    else:
        df = st.session_state['current_df']
        meta = get_dataset_metadata(df, st.session_state['file_name'])
        classified = classify_columns(df)

        # Overview KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{meta["rows"]:,}</div><div class="metric-lbl">Total Rows</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{meta["columns"]}</div><div class="metric-lbl">Total Columns</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(classified["numerical"])}</div><div class="metric-lbl">Numerical Cols</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(classified["categorical"])}</div><div class="metric-lbl">Categorical Cols</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{format_memory_size(meta["memory_bytes"])}</div><div class="metric-lbl">Memory Size</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dataset Preview
        st.subheader("Dataset Preview")
        preview_rows = st.slider("Number of preview rows", min_value=5, max_value=min(100, len(df)), value=min(10, len(df)), step=5)
        st.dataframe(get_preview(df, preview_rows), use_container_width=True)

        # Column Classification Summary
        st.subheader("Dynamic Column Classification")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("**🔢 Numerical Columns:**")
            if classified["numerical"]:
                for c in classified["numerical"]:
                    st.code(c, language="text")
            else:
                st.caption("None detected")
        with col_c2:
            st.markdown("**🔤 Categorical / Text Columns:**")
            if classified["categorical"]:
                for c in classified["categorical"]:
                    st.code(c, language="text")
            else:
                st.caption("None detected")
        with col_c3:
            st.markdown("**📅 Date / Time Columns:**")
            if classified["datetime"]:
                for c in classified["datetime"]:
                    st.code(c, language="text")
            else:
                st.caption("None detected")

        # Detailed Column Schema Table
        st.subheader("Comprehensive Column Schema")
        summary_table = get_column_summary_table(df)
        st.dataframe(summary_table, use_container_width=True)

        # Safe Data Type Converter Tool
        with st.expander("🛠️ Advanced: Safe Column Type Converter", expanded=False):
            st.caption("Convert a column's data type safely without modifying the raw source file.")
            col_sel, type_sel, conv_btn = st.columns([2, 2, 1])
            with col_sel:
                target_col = st.selectbox("Select Column to Convert", df.columns)
            with type_sel:
                target_type = st.selectbox("Convert To", ["Numeric", "Datetime", "String", "Category"])
            with conv_btn:
                st.write("")
                st.write("")
                if st.button("Convert Type"):
                    new_df, ok, msg = safe_convert_column(df, target_col, target_type)
                    if ok:
                        st.session_state['current_df'] = new_df
                        st.session_state['cleaning_logs'].append(msg)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


# =============================================================================
# 3. CLEANING PAGE (Person 2 — Data Quality)
# =============================================================================
elif page == "Cleaning":
    st.markdown('<div class="team-badge">PERSON 2 | OBJECTIVE 2: DATA CLEANING & QUALITY</div>', unsafe_allow_html=True)
    st.title("Data Quality Inspection & Safe Cleaning")

    if st.session_state['current_df'] is None:
        st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload or select an example dataset.")
    else:
        current_df = st.session_state['current_df']
        original_df = st.session_state['original_df']
        quality_rep = get_data_quality_report(current_df)

        # Quality Metrics
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{quality_rep["quality_score"]}%</div><div class="metric-lbl">Data Health Score</div></div>', unsafe_allow_html=True)
        with q2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{quality_rep["total_missing"]}</div><div class="metric-lbl">Total Missing Cells</div></div>', unsafe_allow_html=True)
        with q3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{quality_rep["rows_with_missing"]}</div><div class="metric-lbl">Rows with Missing Data</div></div>', unsafe_allow_html=True)
        with q4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{quality_rep["duplicate_count"]}</div><div class="metric-lbl">Duplicate Rows</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Missing Values & Duplicates Detailed Inspection
        col_miss, col_dup = st.columns([1.2, 1])

        with col_miss:
            st.subheader("Missing Values by Column")
            if quality_rep["total_missing"] > 0:
                st.dataframe(quality_rep["missing_table"], use_container_width=True)
            else:
                st.success("✅ Excellent! No missing values detected in the active dataset.")

        with col_dup:
            st.subheader("Duplicate Records")
            if quality_rep["duplicate_count"] > 0:
                st.warning(f"⚠️ {quality_rep['duplicate_count']} duplicate record(s) found.")
                with st.expander("Preview Duplicate Rows"):
                    st.dataframe(quality_rep["duplicate_rows"], use_container_width=True)
            else:
                st.success("✅ No duplicate records detected in the active dataset.")

        st.markdown("---")

        # Interactive Cleaning Controls
        st.subheader("Safe Cleaning Operations")
        st.caption("Clean the working copy while preserving original data integrity.")

        clean_tab1, clean_tab2, clean_tab3, clean_tab4 = st.tabs([
            "1. Missing Numerical Values",
            "2. Missing Categorical Values",
            "3. Duplicate Records",
            "4. Text Cleanup"
        ])

        # 1. Missing Numerical
        with clean_tab1:
            num_cols = [c for c in current_df.columns if pd.api.types.is_numeric_dtype(current_df[c])]
            num_cols_missing = [c for c in num_cols if current_df[c].isna().sum() > 0]

            if num_cols_missing:
                st.info(f"Numerical columns with missing values: {', '.join(num_cols_missing)}")
                sel_num_cols = st.multiselect("Select Numerical Column(s) to Impute", num_cols_missing, default=num_cols_missing)
                strategy_num = st.selectbox("Imputation Strategy", ["Mean", "Median", "Fill with 0", "Drop rows with missing values"])

                if st.button("Apply Numerical Cleaning", type="primary"):
                    strat_map = {
                        "Mean": "mean",
                        "Median": "median",
                        "Fill with 0": "zero",
                        "Drop rows with missing values": "drop"
                    }
                    cleaned, msg = handle_missing_numerical(current_df, sel_num_cols, strat_map[strategy_num])
                    st.session_state['current_df'] = cleaned
                    st.session_state['cleaning_logs'].append(msg)
                    st.success(msg)
                    st.rerun()
            else:
                st.success("No missing values in numerical columns.")

        # 2. Missing Categorical
        with clean_tab2:
            cat_cols = [c for c in current_df.columns if not pd.api.types.is_numeric_dtype(current_df[c])]
            cat_cols_missing = [c for c in cat_cols if current_df[c].isna().sum() > 0]

            if cat_cols_missing:
                st.info(f"Categorical columns with missing values: {', '.join(cat_cols_missing)}")
                sel_cat_cols = st.multiselect("Select Categorical Column(s) to Handle", cat_cols_missing, default=cat_cols_missing)
                strategy_cat = st.selectbox("Categorical Strategy", ["Fill with Mode (Most Frequent)", "Fill with 'Unknown'", "Drop rows with missing values"])

                if st.button("Apply Categorical Cleaning", type="primary"):
                    strat_map = {
                        "Fill with Mode (Most Frequent)": "mode",
                        "Fill with 'Unknown'": "custom",
                        "Drop rows with missing values": "drop"
                    }
                    cleaned, msg = handle_missing_categorical(current_df, sel_cat_cols, strat_map[strategy_cat], custom_val="Unknown")
                    st.session_state['current_df'] = cleaned
                    st.session_state['cleaning_logs'].append(msg)
                    st.success(msg)
                    st.rerun()
            else:
                st.success("No missing values in categorical columns.")

        # 3. Duplicates
        with clean_tab3:
            if quality_rep["duplicate_count"] > 0:
                st.write(f"Click below to remove all {quality_rep['duplicate_count']} duplicate records:")
                if st.button("Remove Duplicate Rows", type="primary"):
                    cleaned, count_rem = remove_duplicates(current_df)
                    st.session_state['current_df'] = cleaned
                    msg = f"Removed {count_rem} duplicate record(s)."
                    st.session_state['cleaning_logs'].append(msg)
                    st.success(msg)
                    st.rerun()
            else:
                st.success("No duplicate records to remove.")

        # 4. Text Cleanup
        with clean_tab4:
            st.write("Strip extraneous leading and trailing spaces from all string/text columns:")
            if st.button("Trim Whitespace in All Text Columns"):
                cleaned, msg = trim_whitespace(current_df)
                st.session_state['current_df'] = cleaned
                st.session_state['cleaning_logs'].append(msg)
                st.success(msg)
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Before vs After Cleaning Comparison Table
        st.subheader("Data Cleaning Verification: Before vs After")
        comp_df = compare_before_after(original_df, current_df)
        st.table(comp_df)

        # Reset button
        if st.button("🔄 Reset to Original Uploaded Dataset"):
            st.session_state['current_df'] = original_df.copy()
            st.session_state['cleaning_logs'].append("Reset dataset to original uploaded state.")
            st.info("Dataset has been restored to its original state.")
            st.rerun()

        # Audit Log
        if st.session_state['cleaning_logs']:
            with st.expander("📋 Cleaning Action Log", expanded=False):
                for i, entry in enumerate(st.session_state['cleaning_logs'], 1):
                    st.text(f"{i}. {entry}")


# =============================================================================
# 4. EXPLORE PAGE (Person 3 — Search, Filter, Sort)
# =============================================================================
elif page == "Explore":
    st.markdown('<div class="team-badge">PERSON 3 | OBJECTIVE 3: DATA EXPLORATION</div>', unsafe_allow_html=True)
    st.title("Data Exploration: Search, Filter, and Sort")

    if st.session_state['current_df'] is None:
        st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload or select an example dataset.")
    else:
        df = st.session_state['current_df']

        explore_tab1, explore_tab2, explore_tab3 = st.tabs([
            "🔍 Dynamic Search",
            "⚡ Dynamic Filter",
            "🔃 Dynamic Sort"
        ])

        # TAB 1: SEARCH
        with explore_tab1:
            st.subheader("Dynamic Search Across Records")
            s_col1, s_col2, s_col3 = st.columns([1.5, 2, 1])

            with s_col1:
                search_column = st.selectbox("Search In Column", ["__ALL__ (All Columns)"] + list(df.columns), key="search_col")
            
            # Dynamic input depending on column type
            is_num_search = (search_column in df.columns and pd.api.types.is_numeric_dtype(df[search_column]))

            with s_col2:
                if is_num_search:
                    num_op = st.selectbox("Condition", ["==", "!=", ">", ">=", "<", "<=", "Between"], key="search_num_op")
                    if num_op == "Between":
                        search_val = st.text_input("Enter Range (e.g. 50, 90)", key="search_q_num_between")
                    else:
                        search_val = st.text_input(f"Enter numeric value for {num_op}", key="search_q_num")
                else:
                    search_val = st.text_input("Enter search text / keyword", placeholder="e.g. Computer Science, North, etc.", key="search_q_text")
                    num_op = "=="

            with s_col3:
                st.write("")
                st.write("")
                case_sensitive = st.checkbox("Case-sensitive", value=False) if not is_num_search else False
                run_search = st.button("Search Records", type="primary")

            if search_val:
                search_results, search_msg = search_dataset(
                    df,
                    search_column if search_column != "__ALL__ (All Columns)" else "__ALL__",
                    search_val,
                    case_sensitive=case_sensitive,
                    num_operator=num_op
                )
                st.info(search_msg)
                st.dataframe(search_results, use_container_width=True)
            else:
                st.dataframe(get_preview(df, 15), use_container_width=True)
                st.caption(f"Showing initial 15 of {len(df)} records. Enter a search query above to filter.")

        # TAB 2: FILTER
        with explore_tab2:
            st.subheader("Dynamic Column Filtering")
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.2, 1.5, 1])

            with f_col1:
                filter_column = st.selectbox("Select Column to Filter", df.columns, key="filter_col_sel")

            is_filter_num = pd.api.types.is_numeric_dtype(df[filter_column])

            with f_col2:
                if is_filter_num:
                    f_operator = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<=", "Between"], key="filter_op_num")
                else:
                    f_operator = st.selectbox("Operator", ["Equals", "Does not equal", "Contains", "Starts with", "Ends with", "In list"], key="filter_op_cat")

            with f_col3:
                if is_filter_num:
                    if f_operator == "Between":
                        f_v1 = st.number_input("Minimum Value", value=float(df[filter_column].min()), key="f_v1_num")
                        f_v2 = st.number_input("Maximum Value", value=float(df[filter_column].max()), key="f_v2_num")
                    else:
                        f_v1 = st.number_input("Value", value=float(df[filter_column].mean() if not df[filter_column].isna().all() else 0.0), key="f_v_num")
                        f_v2 = None
                else:
                    if f_operator == "In list":
                        f_v1 = st.text_input("Values (comma-separated)", placeholder="item1, item2, item3", key="f_v_list")
                    else:
                        f_v1 = st.text_input("Filter Text", key="f_v_text")
                    f_v2 = None

            with f_col4:
                st.write("")
                st.write("")
                apply_filter = st.button("Apply Filter", type="primary")

            if apply_filter and (f_v1 is not None and str(f_v1) != ""):
                filtered_df, filter_msg = filter_dataset(df, filter_column, f_operator, f_v1, f_v2)
                st.success(filter_msg)
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(get_preview(df, 15), use_container_width=True)
                st.caption(f"Showing initial 15 of {len(df)} records. Select filter criteria and click 'Apply Filter'.")

        # TAB 3: SORT
        with explore_tab3:
            st.subheader("Dynamic Sorting")
            sort_c1, sort_c2, sort_c3 = st.columns([2, 1.5, 1])

            with sort_c1:
                sort_col = st.selectbox("Sort By Column", df.columns, key="sort_col_sel")
            with sort_c2:
                sort_order = st.radio("Order", ["Ascending (Lowest first)", "Descending (Highest first)"], horizontal=True)
            with sort_c3:
                st.write("")
                st.write("")
                sort_btn = st.button("Sort Records", type="primary")

            is_asc = "Ascending" in sort_order
            sorted_df = sort_dataset(df, sort_col, ascending=is_asc)
            st.caption(f"Sorted by '{sort_col}' ({'Ascending' if is_asc else 'Descending'}):")
            st.dataframe(sorted_df, use_container_width=True)


# =============================================================================
# 5. STATISTICS PAGE (Person 3 — Statistical Analysis)
# =============================================================================
elif page == "Statistics":
    st.markdown('<div class="team-badge">PERSON 3 | OBJECTIVE 3: STATISTICAL ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Statistical Analysis & Summaries")

    if st.session_state['current_df'] is None:
        st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload or select an example dataset.")
    else:
        df = st.session_state['current_df']
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        # Section 1: Single Numerical Column Deep-Dive
        st.subheader("Numerical Column Statistics Deep-Dive")
        if num_cols:
            selected_num = st.selectbox("Select Numerical Column", num_cols)
            stats, err = get_single_column_stats(df, selected_num)

            if stats:
                st.markdown(f"**Key Statistics for `{selected_num}`:**")
                sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
                with sc1:
                    st.metric("Count", f"{stats['Count']:,}")
                with sc2:
                    st.metric("Mean", f"{stats['Mean']:.2f}")
                with sc3:
                    st.metric("Median", f"{stats['Median']:.2f}")
                with sc4:
                    st.metric("Std Dev", f"{stats['Standard Deviation']:.2f}")
                with sc5:
                    st.metric("Minimum", f"{stats['Minimum']:.2f}")
                with sc6:
                    st.metric("Maximum", f"{stats['Maximum']:.2f}")

                # Secondary row of metrics
                with st.expander("Additional Statistical Measures"):
                    ac1, ac2, ac3, ac4, ac5 = st.columns(5)
                    with ac1:
                        st.metric("Variance", f"{stats['Variance']:.2f}")
                    with ac2:
                        st.metric("25% (Q1)", f"{stats['25th Percentile (Q1)']:.2f}")
                    with ac3:
                        st.metric("75% (Q3)", f"{stats['75th Percentile (Q3)']:.2f}")
                    with ac4:
                        st.metric("IQR", f"{stats['IQR (Interquartile Range)']:.2f}")
                    with ac5:
                        st.metric("Skewness", f"{stats['Skewness']:.2f}")
            else:
                st.warning(err)
        else:
            st.warning("No numerical columns are available for statistical analysis.")

        st.markdown("---")

        # Section 2: Comprehensive Numerical Summary Table
        st.subheader("Descriptive Summary of All Numerical Columns")
        num_summary_df, num_err = get_all_numerical_summary(df)
        if num_summary_df is not None:
            st.dataframe(num_summary_df, use_container_width=True)
        else:
            st.warning(num_err)

        st.markdown("---")

        # Section 3: Categorical Frequency Breakdown
        st.subheader("Categorical Column Frequency & Mode Analysis")
        if cat_cols:
            selected_cat = st.selectbox("Select Categorical Column", cat_cols)
            cat_sum, cat_vc = get_categorical_summary(df, selected_cat)

            if cat_sum:
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.metric("Unique Categories", cat_sum["Unique Categories"])
                with cc2:
                    st.metric("Most Frequent (Mode)", cat_sum["Most Frequent (Mode)"])
                with cc3:
                    st.metric("Mode Frequency", f"{cat_sum['Mode Frequency']} ({cat_sum['Mode Dominance']})")

                if not cat_vc.empty:
                    st.dataframe(cat_vc, use_container_width=True)
        else:
            st.info("No categorical columns in this dataset.")

        st.markdown("---")

        # Section 4: Correlation Matrix
        st.subheader("Pearson Correlation Matrix (Numerical Columns)")
        corr_matrix, corr_err = get_correlation_matrix(df)
        if corr_matrix is not None:
            st.dataframe(corr_matrix, use_container_width=True)
        else:
            st.info(corr_err)


# =============================================================================
# 6. VISUALIZATION PAGE (Person 4 — Visualization & Charts)
# =============================================================================
elif page == "Visualization":
    st.markdown('<div class="team-badge">PERSON 4 | OBJECTIVE 4: DATA VISUALIZATION</div>', unsafe_allow_html=True)
    st.title("Dynamic Data Visualization")

    if st.session_state['current_df'] is None:
        st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload or select an example dataset.")
    else:
        df = st.session_state['current_df']
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Bar Chart",
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Line Chart",
                "Correlation Heatmap"
            ]
        )

        fig, error_msg = None, None

        # 1. BAR CHART
        if chart_type == "Bar Chart":
            st.caption("Compare category frequencies or aggregated numerical metrics.")
            bc1, bc2, bc3, bc4 = st.columns([1.5, 1.5, 1, 1])
            with bc1:
                bar_cat = st.selectbox("Category Column (X-axis)", cat_cols if cat_cols else df.columns)
            with bc2:
                bar_num = st.selectbox("Aggregate Column (Optional)", ["None (Frequency Count)"] + num_cols)
            with bc3:
                bar_agg = st.selectbox("Aggregation", ["Mean", "Sum"]) if bar_num != "None (Frequency Count)" else "Count"
            with bc4:
                bar_top_n = st.slider("Top N Categories", 5, 30, 10)

            chosen_num = None if bar_num == "None (Frequency Count)" else bar_num
            fig, error_msg = plot_bar_chart(df, bar_cat, chosen_num, bar_agg, bar_top_n)

        # 2. HISTOGRAM
        elif chart_type == "Histogram":
            st.caption("Inspect the frequency distribution of a numerical column.")
            if not num_cols:
                error_msg = "No numerical columns available for Histogram."
            else:
                hc1, hc2, hc3 = st.columns([2, 1.5, 1])
                with hc1:
                    hist_col = st.selectbox("Select Numerical Column", num_cols)
                with hc2:
                    hist_bins = st.slider("Number of Bins", 5, 50, 15)
                with hc3:
                    st.write("")
                    st.write("")
                    hist_kde = st.checkbox("Show KDE Curve", value=True)

                fig, error_msg = plot_histogram(df, hist_col, hist_bins, hist_kde)

        # 3. BOX PLOT
        elif chart_type == "Box Plot":
            st.caption("Analyze statistical spread, quartiles, and outliers.")
            if not num_cols:
                error_msg = "No numerical columns available for Box Plot."
            else:
                bxc1, bxc2 = st.columns(2)
                with bxc1:
                    box_num = st.selectbox("Select Numerical Column", num_cols)
                with bxc2:
                    box_cat = st.selectbox("Group By Category (Optional)", ["None"] + cat_cols)

                chosen_cat = None if box_cat == "None" else box_cat
                fig, error_msg = plot_boxplot(df, box_num, chosen_cat)

        # 4. SCATTER PLOT
        elif chart_type == "Scatter Plot":
            st.caption("Analyze the relationship and correlation between two numerical variables.")
            if len(num_cols) < 2:
                error_msg = "Scatter plots require two numerical columns. Please select suitable columns."
            else:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    scat_x = st.selectbox("X-Axis Numerical Column", num_cols, index=0)
                with sc2:
                    scat_y = st.selectbox("Y-Axis Numerical Column", num_cols, index=min(1, len(num_cols)-1))
                with sc3:
                    scat_hue = st.selectbox("Color / Group By (Optional)", ["None"] + cat_cols)

                chosen_hue = None if scat_hue == "None" else scat_hue
                fig, error_msg = plot_scatterplot(df, scat_x, scat_y, chosen_hue)

        # 5. LINE CHART
        elif chart_type == "Line Chart":
            st.caption("Analyze trends over an ordered sequence, index, or time axis.")
            if not num_cols:
                error_msg = "Line charts require a numerical column for the value axis."
            else:
                lc1, lc2 = st.columns(2)
                with lc1:
                    line_y = st.selectbox("Value Column (Y-axis)", num_cols)
                with lc2:
                    line_x = st.selectbox("Sequence Column (Optional X-axis)", ["None (Record Index)"] + list(df.columns))

                chosen_x = None if line_x == "None (Record Index)" else line_x
                fig, error_msg = plot_line_chart(df, line_y, chosen_x)

        # 6. CORRELATION HEATMAP
        elif chart_type == "Correlation Heatmap":
            st.caption("Visualize pairwise Pearson correlation coefficients across numerical variables.")
            if len(num_cols) < 2:
                error_msg = "Correlation heatmap requires multiple numerical columns."
            else:
                fig, error_msg = plot_correlation_heatmap(df)

        # Display Figure or Validation Message
        if error_msg:
            st.warning(f"⚠️ {error_msg}")
        elif fig is not None:
            st.pyplot(fig)

            # Option to save chart into report session
            if st.button("💾 Save Chart to Final Report"):
                chart_name = f"{chart_type} - {len(st.session_state['saved_charts']) + 1}"
                st.session_state['saved_charts'].append({
                    "title": chart_name,
                    "fig": fig
                })
                st.success(f"Saved '{chart_name}' to final report gallery!")


# =============================================================================
# 7. REPORT & DOWNLOAD PAGE (Person 4 — Key Insights & Report)
# =============================================================================
elif page == "Report":
    st.markdown('<div class="team-badge">PERSON 4 | OBJECTIVE 4: REPORT & DOWNLOAD</div>', unsafe_allow_html=True)
    st.title("Final Analysis Report & Key Insights")

    if st.session_state['current_df'] is None:
        st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload or select an example dataset.")
    else:
        current_df = st.session_state['current_df']
        original_df = st.session_state['original_df']

        # Key Factual Insights Section
        st.subheader("💡 Key Factual Insights")
        st.caption("Factual observations derived strictly from the dataset.")

        insights = generate_factual_insights(current_df, original_df)
        for ins in insights:
            st.markdown(f"""
            <div class="insight-card">
                <strong>{ins['icon']} [{ins['category']}] {ins['headline']}</strong><br>
                <span style="color: #4b5563; font-size: 0.95rem;">{ins['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Dataset & Cleaning Summary
        st.subheader("Data Cleaning Summary")
        comp_df = compare_before_after(original_df, current_df)
        st.table(comp_df)

        if st.session_state['cleaning_logs']:
            st.markdown("**Cleaning Log:**")
            for i, l in enumerate(st.session_state['cleaning_logs'], 1):
                st.text(f"{i}. {l}")

        st.markdown("---")

        # Statistical Summary
        st.subheader("Statistical Summary Table")
        num_summary, s_err = get_all_numerical_summary(current_df)
        if num_summary is not None:
            st.dataframe(num_summary, use_container_width=True)
        else:
            st.info(s_err)

        st.markdown("---")

        # Saved Visualizations Gallery
        if st.session_state['saved_charts']:
            st.subheader("Saved Analysis Visualizations")
            for saved in st.session_state['saved_charts']:
                st.markdown(f"**{saved['title']}**")
                st.pyplot(saved['fig'])
            st.markdown("---")

        # Downloads Section
        st.subheader("📥 Export & Downloads")
        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            cleaned_csv_bytes = get_csv_download_bytes(current_df)
            st.download_button(
                label="📄 Download Cleaned Dataset (.csv)",
                data=cleaned_csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with d_col2:
            report_text = generate_text_report(
                st.session_state['file_name'],
                original_df,
                current_df,
                st.session_state['cleaning_logs']
            )
            st.download_button(
                label="📑 Download Full Report (.txt)",
                data=report_text.encode('utf-8'),
                file_name="analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        with d_col3:
            if num_summary is not None:
                summary_csv = get_csv_download_bytes(num_summary)
                st.download_button(
                    label="📊 Download Statistics (.csv)",
                    data=summary_csv,
                    file_name="statistical_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
