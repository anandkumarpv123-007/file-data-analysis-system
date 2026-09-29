"""
Module 5: Statistical Analysis
==============================
Responsibilities:
- Calculate key statistical measures for numerical columns:
    * Count
    * Mean
    * Median
    * Minimum
    * Maximum
    * Standard Deviation
    * Variance, 25th / 75th Percentiles, Interquartile Range (IQR), Skewness
- Comprehensive numerical summary table across all numerical columns.
- Categorical frequency distribution and mode analysis.
- Correlation matrix computation for numerical columns.
- Robust error handling: never calculate mean on categorical columns,
  return clean error if no numerical columns exist.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


def get_single_column_stats(df: pd.DataFrame, column: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Compute comprehensive statistics for a single numerical column.
    
    Returns:
        (stats_dict, None) on success
        (None, error_message) on failure
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    if column not in df.columns:
        return None, f"Column '{column}' does not exist."

    series = df[column].dropna()

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, f"Column '{column}' is categorical. Numerical statistics (mean, median, std) cannot be calculated on text columns."

    if len(series) == 0:
        return None, f"Column '{column}' contains no non-null numeric values."

    cnt = int(series.count())
    mean_val = float(series.mean())
    median_val = float(series.median())
    min_val = float(series.min())
    max_val = float(series.max())
    std_val = float(series.std(ddof=1)) if cnt > 1 else 0.0
    var_val = float(series.var(ddof=1)) if cnt > 1 else 0.0
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    iqr_val = float(q75 - q25)
    skew_val = float(series.skew()) if cnt > 2 else 0.0

    stats = {
        "Column": column,
        "Count": cnt,
        "Mean": round(mean_val, 2),
        "Median": round(median_val, 2),
        "Minimum": round(min_val, 2),
        "Maximum": round(max_val, 2),
        "Standard Deviation": round(std_val, 2),
        "Variance": round(var_val, 2),
        "25th Percentile (Q1)": round(q25, 2),
        "75th Percentile (Q3)": round(q75, 2),
        "IQR (Interquartile Range)": round(iqr_val, 2),
        "Skewness": round(skew_val, 2)
    }

    return stats, None


def get_all_numerical_summary(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Generate an enhanced descriptive summary table for all numerical columns.
    Includes: Count, Mean, Median, Std, Min, 25%, 50% (Median), 75%, Max, IQR.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        return None, "No numerical columns are available for statistical analysis."

    records = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        cnt = int(s.count())
        mean_v = round(float(s.mean()), 2)
        med_v = round(float(s.median()), 2)
        min_v = round(float(s.min()), 2)
        max_v = round(float(s.max()), 2)
        std_v = round(float(s.std(ddof=1)), 2) if cnt > 1 else 0.0
        q25 = round(float(s.quantile(0.25)), 2)
        q75 = round(float(s.quantile(0.75)), 2)
        iqr = round(q75 - q25, 2)

        records.append({
            "Column": col,
            "Count": cnt,
            "Mean": mean_v,
            "Median": med_v,
            "Min": min_v,
            "25%": q25,
            "50%": med_v,
            "75%": q75,
            "Max": max_v,
            "Std Dev": std_v,
            "IQR": iqr
        })

    if not records:
        return None, "All numerical columns contain only NaN values."

    return pd.DataFrame(records), None


def get_categorical_summary(df: pd.DataFrame, column: str) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Get frequency table and summary metrics for a categorical column.
    """
    if df is None or df.empty or column not in df.columns:
        return None, None

    series = df[column].dropna()
    total = len(series)
    unique_cnt = series.nunique()

    if total == 0:
        return {
            "Column": column,
            "Total Non-Null": 0,
            "Unique Categories": 0,
            "Mode": "None",
            "Mode Count": 0,
            "Mode Percentage": "0%"
        }, pd.DataFrame()

    mode_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
    mode_cnt = int((series == mode_val).sum())
    mode_pct = round(mode_cnt / total * 100, 1)

    summary = {
        "Column": column,
        "Total Non-Null": total,
        "Unique Categories": unique_cnt,
        "Most Frequent (Mode)": str(mode_val),
        "Mode Frequency": mode_cnt,
        "Mode Dominance": f"{mode_pct}%"
    }

    # Value counts table
    vc = series.value_counts().reset_index()
    vc.columns = ["Category", "Count"]
    vc["Percentage"] = (vc["Count"] / total * 100).round(1).astype(str) + "%"

    return summary, vc


def get_correlation_matrix(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Compute Pearson correlation matrix for all numerical columns.
    Returns:
        (corr_df, None) if >= 2 numerical columns exist
        (None, error_msg) otherwise
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) < 2:
        return None, "Correlation matrix requires at least two numerical columns."

    corr_df = num_df.corr(method='pearson').round(3)
    return corr_df, None
