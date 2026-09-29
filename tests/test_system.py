"""
Comprehensive Automated Test Suite
==================================
Tests:
- Test A: Student Performance dataset (Education)
- Test B: Sales dataset (Commercial)
- Test C: Employee dataset (HR)
- Test D: Customer tab-delimited text dataset (Retail)
- Test E: Error handling (Empty file, invalid text, missing numeric cols)
- Test F: Data cleaning operations & before/after audit
- Test G: Dynamic Search, Filter, Sort
- Test H: Descriptive & Single-Column Statistics
- Test I: Matplotlib & Seaborn Visualizations + Validation checks
- Test J: Key Factual Insights & Final Report Generation
"""

import os
import io
import unittest
import pandas as pd
import numpy as np

from modules.file_handler import load_dataset, get_dataset_metadata, get_preview
from modules.processing import classify_columns, get_column_summary_table, safe_convert_column
from modules.cleaning import (
    get_data_quality_report,
    remove_duplicates,
    handle_missing_numerical,
    handle_missing_categorical,
    trim_whitespace,
    compare_before_after
)
from modules.exploration import search_dataset, filter_dataset, sort_dataset, paginate_dataframe
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


class TestFileDataAnalysisSystem(unittest.TestCase):

    def setUp(self):
        self.student_csv = os.path.join("data", "student_performance.csv")
        self.sales_csv = os.path.join("data", "sales_data.csv")
        self.employee_csv = os.path.join("data", "employee_data.csv")
        self.customer_txt = os.path.join("data", "customer_data.txt")

    # -------------------------------------------------------------------------
    # TEST SUITE A: STUDENT PERFORMANCE DATASET
    # -------------------------------------------------------------------------
    def test_student_dataset_pipeline(self):
        # 1. Load
        df, err = load_dataset(self.student_csv, "student_performance.csv")
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 30)

        # 2. Metadata & Classification
        meta = get_dataset_metadata(df, "student_performance.csv")
        self.assertEqual(meta["rows"], len(df))
        classified = classify_columns(df)
        self.assertIn("Maths_Score", classified["numerical"])
        self.assertIn("Department", classified["categorical"])

        # 3. Quality & Cleaning
        qual = get_data_quality_report(df)
        self.assertGreater(qual["total_missing"], 0)
        self.assertGreater(qual["duplicate_count"], 0)

        # Remove dups
        cleaned, dups_rem = remove_duplicates(df)
        self.assertEqual(dups_rem, qual["duplicate_count"])

        # Impute missing numerical
        cleaned, msg1 = handle_missing_numerical(cleaned, ["Maths_Score"], strategy="mean")
        self.assertEqual(int(cleaned["Maths_Score"].isna().sum()), 0)

        # Impute missing categorical
        cleaned, msg2 = handle_missing_categorical(cleaned, ["Department"], strategy="mode")
        self.assertEqual(int(cleaned["Department"].isna().sum()), 0)

        # 4. Search & Filter & Sort
        res_search, s_msg = search_dataset(cleaned, "Department", "Computer Science")
        self.assertGreater(len(res_search), 0)

        res_filter, f_msg = filter_dataset(cleaned, "Maths_Score", ">=", 80)
        self.assertGreater(len(res_filter), 0)

        res_sorted = sort_dataset(cleaned, "Maths_Score", ascending=False)
        self.assertGreaterEqual(res_sorted["Maths_Score"].iloc[0], res_sorted["Maths_Score"].iloc[-1])

        # 5. Statistics
        stats, stat_err = get_single_column_stats(cleaned, "Maths_Score")
        self.assertIsNone(stat_err)
        self.assertIn("Mean", stats)
        self.assertIn("Median", stats)
        self.assertIn("Standard Deviation", stats)

        num_summary, n_err = get_all_numerical_summary(cleaned)
        self.assertIsNone(n_err)
        self.assertGreaterEqual(len(num_summary), 4)

        # 6. Visualizations
        fig_bar, b_err = plot_bar_chart(cleaned, "Department", "Maths_Score", "Mean")
        self.assertIsNone(b_err)
        self.assertIsNotNone(fig_bar)

        fig_hist, h_err = plot_histogram(cleaned, "Maths_Score")
        self.assertIsNone(h_err)
        self.assertIsNotNone(fig_hist)

        fig_box, bx_err = plot_boxplot(cleaned, "Maths_Score", "Department")
        self.assertIsNone(bx_err)
        self.assertIsNotNone(fig_box)

        fig_scat, sc_err = plot_scatterplot(cleaned, "Maths_Score", "Python_Score")
        self.assertIsNone(sc_err)
        self.assertIsNotNone(fig_scat)

        fig_heat, ht_err = plot_correlation_heatmap(cleaned)
        self.assertIsNone(ht_err)
        self.assertIsNotNone(fig_heat)

        # 7. Insights & Reporting
        insights = generate_factual_insights(cleaned, df)
        self.assertGreater(len(insights), 3)

        report = generate_text_report("student_performance.csv", df, cleaned, [msg1, msg2])
        self.assertIn("FILE DATA ANALYSIS SYSTEM", report)
        self.assertIn("student_performance.csv", report)

    # -------------------------------------------------------------------------
    # TEST SUITE B: SALES DATASET (COMMERCIAL DOMAIN)
    # -------------------------------------------------------------------------
    def test_sales_dataset_pipeline(self):
        df, err = load_dataset(self.sales_csv, "sales_data.csv")
        self.assertIsNone(err)
        classified = classify_columns(df)
        self.assertIn("Sales_Amount", classified["numerical"])
        self.assertIn("Region", classified["categorical"])

        stats, stat_err = get_single_column_stats(df, "Sales_Amount")
        self.assertIsNone(stat_err)
        self.assertGreater(stats["Mean"], 0)

        # Test Date / Region / Aggregation
        fig, b_err = plot_bar_chart(df, "Region", "Sales_Amount", "Sum")
        self.assertIsNone(b_err)
        self.assertIsNotNone(fig)

        fig_line, l_err = plot_line_chart(df, "Sales_Amount", "Order_Date")
        self.assertIsNone(l_err)
        self.assertIsNotNone(fig_line)

    # -------------------------------------------------------------------------
    # TEST SUITE C: EMPLOYEE DATASET (HR DOMAIN)
    # -------------------------------------------------------------------------
    def test_employee_dataset_pipeline(self):
        df, err = load_dataset(self.employee_csv, "employee_data.csv")
        self.assertIsNone(err)
        classified = classify_columns(df)
        self.assertIn("Salary", classified["numerical"])
        self.assertIn("Department", classified["categorical"])

        summary, s_err = get_all_numerical_summary(df)
        self.assertIsNone(s_err)
        self.assertGreaterEqual(len(summary), 3)

        corr, c_err = get_correlation_matrix(df)
        self.assertIsNone(c_err)
        self.assertIsNotNone(corr)

    # -------------------------------------------------------------------------
    # TEST SUITE D: CUSTOMER DATASET (TAB-DELIMITED TXT)
    # -------------------------------------------------------------------------
    def test_customer_txt_pipeline(self):
        df, err = load_dataset(self.customer_txt, "customer_data.txt")
        self.assertIsNone(err)
        self.assertEqual(len(df.columns), 6)
        self.assertIn("Customer_Name", df.columns)
        self.assertIn("Annual_Spend", df.columns)

    # -------------------------------------------------------------------------
    # TEST SUITE E: ERROR HANDLING & EDGE CASES
    # -------------------------------------------------------------------------
    def test_empty_file_handling(self):
        empty_data = b""
        df, err = load_dataset(empty_data, "empty.csv")
        self.assertIsNone(df)
        self.assertEqual(err, "The uploaded file is empty.")

    def test_invalid_tabular_text(self):
        invalid_bytes = b"This is just random text with no columns or structure.\nSecond row with nothing."
        # Even if parsed, single column check
        df, err = load_dataset(invalid_bytes, "invalid.txt")
        # Should either return dataframe or clean message without crashing
        self.assertTrue(df is not None or err is not None)

    def test_visualization_validation_never_crashes(self):
        # Create a DataFrame with categorical columns only
        single_df = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"], "City": ["Delhi", "Mumbai", "Pune"]})

        # Scatter plot on two categorical columns should gracefully return error message instead of crashing
        fig_scat, sc_err = plot_scatterplot(single_df, "Name", "City")
        self.assertIsNone(fig_scat)
        self.assertIn("require two numerical columns", sc_err)

        # Correlation heatmap should gracefully return helpful error message
        fig_heat, ht_err = plot_correlation_heatmap(single_df)
        self.assertIsNone(fig_heat)
        self.assertIn("requires multiple numerical columns", ht_err)

        # Histogram on text column should gracefully reject
        fig_hist, h_err = plot_histogram(single_df, "Name")
        self.assertIsNone(fig_hist)
        self.assertIn("requires a numerical column", h_err)

    def test_statistical_validation_on_categorical(self):
        cat_df = pd.DataFrame({"Category": ["A", "B", "A", "C"]})
        stats, err = get_single_column_stats(cat_df, "Category")
        self.assertIsNone(stats)
        self.assertIn("categorical", err.lower())

        summary, s_err = get_all_numerical_summary(cat_df)
        self.assertIsNone(summary)
        self.assertIn("no numerical columns are available", s_err.lower())

    def test_downloads(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        csv_bytes = get_csv_download_bytes(df)
        self.assertGreater(len(csv_bytes), 0)
        self.assertIn(b"A,B", csv_bytes)


if __name__ == "__main__":
    unittest.main()
