"""
Module 5: Data Visualization & Reporting
=========================================
Responsibilities:
- Dynamic chart generation using Matplotlib and Seaborn.
- Supported Chart Types:
    * Bar Chart (Categorical counts or Categorical vs Numerical aggregation)
    * Histogram (Distribution of numerical column with bins and KDE)
    * Box Plot (Spread, outliers, optional grouping by category)
    * Scatter Plot (Two numerical columns, optional categorical color)
    * Line Chart (Sequential or time-series trend)
    * Correlation Heatmap (Seaborn heatmap of numerical correlation matrix)
- Fail-safe validation: NEVER crash on unsuitable inputs.
- Clean styling, titles, labels, rotation for readability, proper memory cleanup.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend safe for web servers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List


# Set modern clean styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_bar_chart(
    df: pd.DataFrame,
    cat_col: str,
    num_col: Optional[str] = None,
    agg_func: str = "Mean",
    top_n: int = 15
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generate a clean Bar Chart.
    - If num_col is None: plots frequency count of top_n categories.
    - If num_col is provided: plots aggregated metric (Mean or Sum) of num_col grouped by cat_col.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    if cat_col not in df.columns:
        return None, f"Category column '{cat_col}' not found."

    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        if num_col and num_col in df.columns and pd.api.types.is_numeric_dtype(df[num_col]):
            # Aggregated bar chart
            agg_type = 'mean' if agg_func.lower() == 'mean' else 'sum'
            grouped = df.groupby(cat_col)[num_col].agg(agg_type).dropna()
            grouped = grouped.sort_values(ascending=False).head(top_n)

            if grouped.empty:
                plt.close(fig)
                return None, f"No valid data available to aggregate '{num_col}' by '{cat_col}'."

            bars = ax.bar(grouped.index.astype(str), grouped.values, color='#3b82f6', edgecolor='#1e40af')
            ax.set_ylabel(f"{agg_func} of {num_col}", fontsize=11, fontweight='bold')
            ax.set_title(f"{agg_func} of {num_col} by {cat_col} (Top {len(grouped)})", fontsize=13, fontweight='bold', pad=12)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        else:
            # Frequency count bar chart
            counts = df[cat_col].value_counts().head(top_n)
            if counts.empty:
                plt.close(fig)
                return None, f"No data found in '{cat_col}'."

            bars = ax.bar(counts.index.astype(str), counts.values, color='#0ea5e9', edgecolor='#0284c7')
            ax.set_ylabel("Frequency Count", fontsize=11, fontweight='bold')
            ax.set_title(f"Distribution of {cat_col} (Top {len(counts)})", fontsize=13, fontweight='bold', pad=12)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        ax.set_xlabel(cat_col, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
        plt.tight_layout()
        return fig, None

    except Exception as ex:
        if 'fig' in locals():
            plt.close(fig)
        return None, f"Error generating Bar Chart: {str(ex)}"


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 15,
    kde: bool = True
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generate a Histogram with optional KDE curve.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    if column not in df.columns:
        return None, f"Column '{column}' not found."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, f"Histogram requires a numerical column. '{column}' is categorical."

    series = df[column].dropna()
    if len(series) == 0:
        return None, f"Column '{column}' contains only missing values."

    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(series, bins=bins, kde=kde, color='#6366f1', ax=ax, edgecolor='white')

        mean_val = float(series.mean())
        median_val = float(series.median())
        ax.axvline(mean_val, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#10b981', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')

        ax.set_title(f"Histogram of {column}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(column, fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        return fig, None

    except Exception as ex:
        if 'fig' in locals():
            plt.close(fig)
        return None, f"Error generating Histogram: {str(ex)}"


def plot_boxplot(
    df: pd.DataFrame,
    num_col: str,
    cat_col: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generate a Box Plot for outlier and spread inspection.
    Optionally grouped by a categorical column.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    if num_col not in df.columns:
        return None, f"Column '{num_col}' not found."

    if not pd.api.types.is_numeric_dtype(df[num_col]):
        return None, f"Box Plot requires a numerical column. '{num_col}' is categorical."

    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        if cat_col and cat_col in df.columns:
            # Check unique values in cat_col to avoid too many crowded boxes
            unique_cats = df[cat_col].dropna().unique()
            if len(unique_cats) > 12:
                top_cats = df[cat_col].value_counts().head(10).index
                filtered_df = df[df[cat_col].isin(top_cats)]
            else:
                filtered_df = df

            sns.boxplot(
                data=filtered_df,
                x=cat_col,
                y=num_col,
                hue=cat_col,
                legend=False,
                ax=ax,
                palette="Set2"
            )
            ax.set_title(f"Box Plot of {num_col} by {cat_col}", fontsize=13, fontweight='bold', pad=12)
            ax.set_xlabel(cat_col, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=35)
        else:
            sns.boxplot(
                y=df[num_col],
                ax=ax,
                color='#8b5cf6',
                width=0.4
            )
            ax.set_title(f"Box Plot of {num_col}", fontsize=13, fontweight='bold', pad=12)

        ax.set_ylabel(num_col, fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig, None

    except Exception as ex:
        if 'fig' in locals():
            plt.close(fig)
        return None, f"Error generating Box Plot: {str(ex)}"


def plot_scatterplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generate a Scatter Plot for two numerical columns.
    Validates that both columns are numerical.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    # Validation: user must provide two columns
    if not x_col or not y_col:
        return None, "Scatter plots require two numerical columns. Please select suitable columns."

    if x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found in dataset."

    is_x_num = pd.api.types.is_numeric_dtype(df[x_col])
    is_y_num = pd.api.types.is_numeric_dtype(df[y_col])

    if not is_x_num or not is_y_num:
        return None, "Scatter plots require two numerical columns. Please select suitable columns."

    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        clean_sub = df[[x_col, y_col] + ([hue_col] if hue_col and hue_col in df.columns else [])].dropna()

        if clean_sub.empty:
            plt.close(fig)
            return None, "No valid data pairs found without missing values."

        if hue_col and hue_col in df.columns:
            sns.scatterplot(
                data=clean_sub,
                x=x_col,
                y=y_col,
                hue=hue_col,
                palette="tab10",
                alpha=0.85,
                s=60,
                ax=ax
            )
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        else:
            sns.scatterplot(
                data=clean_sub,
                x=x_col,
                y=y_col,
                color='#2563eb',
                alpha=0.8,
                s=60,
                ax=ax
            )

        # Plot trend line if sufficient data points
        if len(clean_sub) > 2:
            try:
                x_vals = clean_sub[x_col].astype(float)
                y_vals = clean_sub[y_col].astype(float)
                m, b = np.polyfit(x_vals, y_vals, 1)
                ax.plot(x_vals, m * x_vals + b, color='#ef4444', linestyle=':', linewidth=1.8, label=f'Trendline (slope={m:.2f})')
                ax.legend(loc='best', frameon=True)
            except Exception:
                pass

        # Correlation note
        corr_val = clean_sub[x_col].corr(clean_sub[y_col])
        ax.set_title(f"{x_col} vs {y_col} (Correlation: {corr_val:.2f})", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig, None

    except Exception as ex:
        if 'fig' in locals():
            plt.close(fig)
        return None, f"Error generating Scatter Plot: {str(ex)}"


def plot_line_chart(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generate a Line Chart for ordered or time-series data.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    if y_col not in df.columns:
        return None, f"Column '{y_col}' not found."

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        return None, f"Line chart value axis requires a numerical column. '{y_col}' is categorical."

    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        if x_col and x_col in df.columns:
            plot_df = df[[x_col, y_col]].dropna().sort_values(by=x_col)
            ax.plot(plot_df[x_col].astype(str), plot_df[y_col], marker='o', markersize=4, color='#0284c7', linewidth=1.8)
            ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=40)
            # Reduce ticks if too many
            if len(plot_df) > 20:
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            ax.set_title(f"Line Chart of {y_col} across {x_col}", fontsize=13, fontweight='bold', pad=12)
        else:
            ax.plot(df.index, df[y_col], color='#0284c7', linewidth=1.8)
            ax.set_xlabel("Record Index", fontsize=11, fontweight='bold')
            ax.set_title(f"Line Chart of {y_col} by Record Sequence", fontsize=13, fontweight='bold', pad=12)

        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig, None

    except Exception as ex:
        if 'fig' in locals():
            plt.close(fig)
        return None, f"Error generating Line Chart: {str(ex)}"


def plot_correlation_heatmap(
    df: pd.DataFrame
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Generate Seaborn Correlation Heatmap for all numerical columns.
    Validates that at least 2 numerical columns exist.
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Validation rule
    if len(num_cols) < 2:
        return None, "Correlation heatmap requires multiple numerical columns."

    try:
        corr_matrix = df[num_cols].corr(method='pearson')

        # Size adapts dynamically to number of columns
        n_cols = len(num_cols)
        figsize = (max(7, n_cols * 1.2), max(5, n_cols * 1.0))
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
            ax=ax
        )
        ax.set_title("Correlation Heatmap (Numerical Variables)", fontsize=13, fontweight='bold', pad=14)
        plt.tight_layout()
        return fig, None

    except Exception as ex:
        if 'fig' in locals():
            plt.close(fig)
        return None, f"Error generating Correlation Heatmap: {str(ex)}"
