"""
Module 4: Data Exploration (Search, Filter, Sort)
=================================================
Responsibilities:
- Dynamic Search:
    * Categorical/Text: case-insensitive substring search.
    * Numerical: sensible numerical conditions (=, >, >=, <, <=, between) or exact match.
    * Global search across all columns.
- Dynamic Filter:
    * Numerical operators: ==, !=, >, >=, <, <=, Between.
    * Categorical operators: Equals, Contains, Starts with, Ends with, In list.
    * Datetime operators: After, Before, Between.
- Dynamic Sort:
    * Any column, Ascending / Descending, NaNs positioning.
- Pagination & display helpers to maintain performance on larger datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any


def search_dataset(
    df: pd.DataFrame,
    column: str,
    query: str,
    case_sensitive: bool = False,
    num_operator: str = "=="
) -> Tuple[pd.DataFrame, str]:
    """
    Search records dynamically in a specified column or across all columns.
    
    If column is '__ALL__': searches across all columns as string.
    If column is numerical: parses query as float and applies num_operator.
    If column is categorical: does case-insensitive contains.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "Dataset is empty."

    if not query or str(query).strip() == "":
        return df, "Showing all records (empty query)."

    query_str = str(query).strip()

    try:
        if column in ["__ALL__", "__ALL__ (All Columns)"]:
            # Global search across all columns converted to string
            mask = pd.Series(False, index=df.index)
            for col in df.columns:
                col_mask = df[col].astype(str).str.contains(query_str, case=case_sensitive, na=False)
                mask = mask | col_mask
            results = df[mask]
            return results, f"Found {len(results)} matching record(s) across all columns."

        if column not in df.columns:
            return df, f"Column '{column}' does not exist in dataset."

        series = df[column]

        # Numerical column handling
        if pd.api.types.is_numeric_dtype(series):
            try:
                # Handle Between format e.g. "10,50" or "10-50"
                if num_operator == "Between":
                    parts = query_str.replace("and", ",").replace("-", ",").split(",")
                    if len(parts) >= 2:
                        val_min = float(parts[0].strip())
                        val_max = float(parts[1].strip())
                        mask = (series >= val_min) & (series <= val_max)
                        results = df[mask]
                        return results, f"Found {len(results)} record(s) where {column} is between {val_min} and {val_max}."
                    else:
                        return df, "For 'Between', enter two numbers separated by a comma (e.g. 10, 50)."

                val_num = float(query_str)
                if num_operator == "==":
                    mask = (series == val_num)
                elif num_operator == "!=":
                    mask = (series != val_num)
                elif num_operator == ">":
                    mask = (series > val_num)
                elif num_operator == ">=":
                    mask = (series >= val_num)
                elif num_operator == "<":
                    mask = (series < val_num)
                elif num_operator == "<=":
                    mask = (series <= val_num)
                else:
                    mask = (series == val_num)

                results = df[mask]
                return results, f"Found {len(results)} record(s) where {column} {num_operator} {val_num}."

            except ValueError:
                # Fallback to string search if query couldn't be parsed as a float
                mask = series.astype(str).str.contains(query_str, case=case_sensitive, na=False)
                results = df[mask]
                return results, f"Found {len(results)} record(s) matching '{query_str}' as text."

        # Categorical / string search
        mask = series.astype(str).str.contains(query_str, case=case_sensitive, na=False)
        results = df[mask]
        return results, f"Found {len(results)} matching record(s) in '{column}'."

    except Exception as ex:
        return df, f"Search error: {str(ex)}"


def filter_dataset(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
    value_secondary: Optional[Any] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Filter DataFrame dynamically based on column, operator, and value.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "Dataset is empty."

    if column not in df.columns:
        return df, f"Column '{column}' not found."

    series = df[column]

    try:
        # 1. Numerical operations
        if pd.api.types.is_numeric_dtype(series):
            val1 = float(value)
            if operator == "==":
                filtered = df[series == val1]
            elif operator == "!=":
                filtered = df[series != val1]
            elif operator == ">":
                filtered = df[series > val1]
            elif operator == ">=":
                filtered = df[series >= val1]
            elif operator == "<":
                filtered = df[series < val1]
            elif operator == "<=":
                filtered = df[series <= val1]
            elif operator == "Between":
                val2 = float(value_secondary) if value_secondary is not None else val1
                min_v, max_v = min(val1, val2), max(val1, val2)
                filtered = df[(series >= min_v) & (series <= max_v)]
            else:
                filtered = df[series == val1]

            pct = round(len(filtered) / len(df) * 100, 1)
            return filtered, f"Filtered {len(filtered)} out of {len(df)} records ({pct}%)."

        # 2. Categorical / string operations
        val_str = str(value).strip()
        str_series = series.astype(str)

        if operator == "Equals":
            filtered = df[str_series.str.lower() == val_str.lower()]
        elif operator == "Does not equal":
            filtered = df[str_series.str.lower() != val_str.lower()]
        elif operator == "Contains":
            filtered = df[str_series.str.contains(val_str, case=False, na=False)]
        elif operator == "Starts with":
            filtered = df[str_series.str.startswith(val_str, na=False)]
        elif operator == "Ends with":
            filtered = df[str_series.str.endswith(val_str, na=False)]
        elif operator == "In list":
            # Comma-separated list of items
            items = [item.strip().lower() for item in val_str.split(",") if item.strip()]
            filtered = df[str_series.str.lower().isin(items)]
        else:
            filtered = df[str_series.str.contains(val_str, case=False, na=False)]

        pct = round(len(filtered) / len(df) * 100, 1)
        return filtered, f"Filtered {len(filtered)} out of {len(df)} records ({pct}%)."

    except Exception as ex:
        return df, f"Filter error: {str(ex)}"


def sort_dataset(
    df: pd.DataFrame,
    column: str,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Sort DataFrame by selected column.
    """
    if df is None or df.empty or column not in df.columns:
        return df

    return df.sort_values(by=column, ascending=ascending, na_position='last').reset_index(drop=True)


def paginate_dataframe(
    df: pd.DataFrame,
    page_number: int = 1,
    page_size: int = 15
) -> Tuple[pd.DataFrame, int, int]:
    """
    Paginate DataFrame for smooth, performant rendering in UI.
    Returns:
        (paginated_df, total_pages, current_page)
    """
    if df is None or df.empty:
        return pd.DataFrame(), 1, 1

    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page_number = max(1, min(page_number, total_pages))

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    return df.iloc[start_idx:end_idx], total_pages, page_number
