"""
Module 2: Data Processing
=========================
Responsibilities:
- Structure inspection (rows, columns, memory usage, shape).
- Dynamic column classification:
    * Numerical columns
    * Categorical/Text columns
    * Date/Time columns (safely detected)
    * Boolean columns
- Safe data type conversions (numeric, datetime, categorical).
- Summary table preparation for user display.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Dynamically classify DataFrame columns into:
    - numerical: integers and floats
    - categorical: text/object, categorical, bool
    - datetime: parsed or potential datetime strings
    - boolean: bool dtypes or binary boolean representations
    """
    if df is None or df.empty:
        return {"numerical": [], "categorical": [], "datetime": [], "boolean": []}

    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    boolean_cols = []

    for col in df.columns:
        series = df[col]

        # 1. Check for boolean
        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
            categorical_cols.append(col)
            continue

        # 2. Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it might actually be a binary flag (0 and 1)
            non_null = series.dropna()
            if len(non_null) > 0 and set(non_null.unique()).issubset({0, 1}):
                numerical_cols.append(col)
            else:
                numerical_cols.append(col)
            continue

        # 3. Check for datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # 4. For string/object columns: test if column might be a date column
        sample = series.dropna().astype(str).head(30)
        if len(sample) > 0:
            is_date = False
            sample_first = sample.iloc[0].strip()
            # Fast heuristic check before expensive parse
            if any(sep in sample_first for sep in ['-', '/', ':']) and len(sample_first) >= 6:
                try:
                    pd.to_datetime(sample, errors='raise')
                    is_date = True
                except (ValueError, TypeError, OverflowError):
                    is_date = False

            if is_date:
                datetime_cols.append(col)
                continue

        # Default to categorical / text
        categorical_cols.append(col)

    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "boolean": boolean_cols
    }


def get_column_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an informative column summary table showing:
    - Column Name
    - Pandas Dtype
    - Detected Role (Numerical / Categorical / Datetime / Boolean)
    - Non-Null Count
    - Missing Count
    - Missing Percentage
    - Unique Values
    """
    if df is None or df.empty:
        return pd.DataFrame()

    classified = classify_columns(df)
    total_rows = len(df)
    records = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        non_null_count = total_rows - null_count
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        if col in classified["datetime"]:
            detected_role = "Date / Time"
        elif col in classified["numerical"]:
            detected_role = "Numerical"
        elif col in classified["boolean"]:
            detected_role = "Boolean"
        else:
            detected_role = "Categorical / Text"

        records.append({
            "Column": col,
            "Detected Role": detected_role,
            "Pandas Dtype": str(series.dtype),
            "Non-Null Count": non_null_count,
            "Missing Count": null_count,
            "Missing %": f"{null_pct}%",
            "Unique Values": unique_count
        })

    return pd.DataFrame(records)


def format_memory_size(bytes_size: int) -> str:
    """Format memory bytes into human-readable string (KB, MB)."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.2f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.2f} MB"


def safe_convert_column(
    df: pd.DataFrame,
    column: str,
    target_type: str
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Safely convert a column's data type without corrupting the DataFrame.
    Target types: 'Numeric', 'Datetime', 'String', 'Category'.
    
    Returns:
        (new_df, success_boolean, status_message)
    """
    if column not in df.columns:
        return df, False, f"Column '{column}' does not exist."

    df_copy = df.copy()

    try:
        if target_type == "Numeric":
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            msg = f"Converted '{column}' to Numeric (unparseable values converted to NaN)."
        elif target_type == "Datetime":
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
            msg = f"Converted '{column}' to Datetime."
        elif target_type == "String":
            df_copy[column] = df_copy[column].astype(str)
            msg = f"Converted '{column}' to String."
        elif target_type == "Category":
            df_copy[column] = df_copy[column].astype('category')
            msg = f"Converted '{column}' to Categorical."
        else:
            return df, False, f"Unknown target type '{target_type}'."

        return df_copy, True, msg

    except Exception as ex:
        return df, False, f"Conversion failed: {str(ex)}"
