"""
Module 3: Data Cleaning
=======================
Responsibilities:
- Detect missing values (per column and total).
- Detect duplicate records.
- Preserve original dataset integrity:
    Original Dataset -> Working Copy -> Cleaning -> Cleaned Dataset
- Provide safe cleaning operations:
    * Fill missing numerical (mean, median, constant 0)
    * Fill missing categorical (mode, constant 'Unknown')
    * Drop rows with missing values (selected or all columns)
    * Remove duplicate records
    * Strip leading/trailing whitespaces in text columns
- Calculate and display Before vs After cleaning metrics.
- Track cleaning action logs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive data quality indicators:
    - Total rows and columns
    - Total missing cells
    - Rows with at least one missing value
    - Missing values per column (DataFrame)
    - Total duplicate rows
    - Duplicate rows DataFrame (head 5)
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_cols": 0,
            "total_missing": 0,
            "rows_with_missing": 0,
            "missing_table": pd.DataFrame(),
            "duplicate_count": 0,
            "duplicate_rows": pd.DataFrame(),
            "quality_score": 100.0
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    # Missing values
    missing_series = df.isna().sum()
    total_missing = int(missing_series.sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())

    # Build missing values table (only columns with missing values or all columns)
    missing_data = []
    for col in df.columns:
        cnt = int(missing_series[col])
        pct = round((cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_data.append({
            "Column": col,
            "Missing Values": cnt,
            "Percentage": f"{pct}%",
            "Dtype": str(df[col].dtype)
        })
    missing_df = pd.DataFrame(missing_data)
    # Sort with highest missing on top
    missing_df = missing_df.sort_values(by="Missing Values", ascending=False).reset_index(drop=True)

    # Duplicates
    dup_mask = df.duplicated(keep='first')
    duplicate_count = int(dup_mask.sum())
    duplicate_rows = df[dup_mask].head(10) if duplicate_count > 0 else pd.DataFrame()

    # Simple Quality Score: 100 - (missing_pct * 0.5 + duplicate_pct * 0.5)
    missing_cell_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    duplicate_row_pct = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
    quality_score = max(0.0, round(100.0 - (missing_cell_pct * 0.7 + duplicate_row_pct * 0.3), 1))

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_missing": total_missing,
        "rows_with_missing": rows_with_missing,
        "missing_table": missing_df,
        "duplicate_count": duplicate_count,
        "duplicate_rows": duplicate_rows,
        "quality_score": quality_score
    }


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from DataFrame.
    Returns:
        (cleaned_df, number_of_duplicates_removed)
    """
    if df is None or df.empty:
        return df, 0

    initial_count = len(df)
    cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed_count = initial_count - len(cleaned)
    return cleaned, removed_count


def handle_missing_numerical(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    custom_val: float = 0.0
) -> Tuple[pd.DataFrame, str]:
    """
    Handle missing values in numerical columns.
    strategy: 'mean', 'median', 'zero', 'custom', 'drop'
    """
    if df is None or df.empty or not columns:
        return df, "No columns specified or DataFrame is empty."

    cleaned = df.copy()
    valid_cols = [c for c in columns if c in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[c])]

    if not valid_cols:
        return df, "No valid numerical columns selected."

    if strategy == "drop":
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=valid_cols).reset_index(drop=True)
        dropped = before - len(cleaned)
        return cleaned, f"Dropped {dropped} row(s) containing missing values in {valid_cols}."

    for col in valid_cols:
        if strategy == "mean":
            val = cleaned[col].mean()
            cleaned[col] = cleaned[col].fillna(val)
        elif strategy == "median":
            val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(val)
        elif strategy == "zero":
            cleaned[col] = cleaned[col].fillna(0)
        elif strategy == "custom":
            cleaned[col] = cleaned[col].fillna(custom_val)

    return cleaned, f"Filled missing values in {valid_cols} using strategy: '{strategy}'."


def handle_missing_categorical(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mode",
    custom_val: str = "Unknown"
) -> Tuple[pd.DataFrame, str]:
    """
    Handle missing values in categorical/text columns.
    strategy: 'mode', 'custom', 'drop'
    """
    if df is None or df.empty or not columns:
        return df, "No columns specified or DataFrame is empty."

    cleaned = df.copy()
    valid_cols = [c for c in columns if c in cleaned.columns]

    if not valid_cols:
        return df, "No valid columns selected."

    if strategy == "drop":
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=valid_cols).reset_index(drop=True)
        dropped = before - len(cleaned)
        return cleaned, f"Dropped {dropped} row(s) containing missing values in {valid_cols}."

    for col in valid_cols:
        if strategy == "mode":
            mode_vals = cleaned[col].mode(dropna=True)
            if not mode_vals.empty:
                cleaned[col] = cleaned[col].fillna(mode_vals[0])
            else:
                cleaned[col] = cleaned[col].fillna(custom_val)
        elif strategy == "custom":
            cleaned[col] = cleaned[col].fillna(custom_val)

    return cleaned, f"Filled missing values in {valid_cols} using strategy: '{strategy}'."


def trim_whitespace(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Strip leading and trailing whitespace from all string/object columns."""
    if df is None or df.empty:
        return df, "DataFrame is empty."

    cleaned = df.copy()
    obj_cols = cleaned.select_dtypes(include=['object', 'string']).columns
    for c in obj_cols:
        cleaned[c] = cleaned[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return cleaned, f"Trimmed whitespace in {len(obj_cols)} text column(s)."


def compare_before_after(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate side-by-side comparison table between original and cleaned dataset.
    """
    before_rows = len(before_df) if before_df is not None else 0
    after_rows = len(after_df) if after_df is not None else 0

    before_missing = int(before_df.isna().sum().sum()) if before_df is not None else 0
    after_missing = int(after_df.isna().sum().sum()) if after_df is not None else 0

    before_dup = int(before_df.duplicated().sum()) if before_df is not None else 0
    after_dup = int(after_df.duplicated().sum()) if after_df is not None else 0

    metrics = [
        {
            "Metric": "Total Rows",
            "Before Cleaning": before_rows,
            "After Cleaning": after_rows,
            "Difference": after_rows - before_rows
        },
        {
            "Metric": "Missing Values (Cells)",
            "Before Cleaning": before_missing,
            "After Cleaning": after_missing,
            "Difference": after_missing - before_missing
        },
        {
            "Metric": "Duplicate Records",
            "Before Cleaning": before_dup,
            "After Cleaning": after_dup,
            "Difference": after_dup - before_dup
        }
    ]
    return pd.DataFrame(metrics)
