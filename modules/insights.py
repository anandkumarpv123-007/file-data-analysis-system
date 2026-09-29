"""
Module: Key Insights Engine
===========================
Responsibilities:
- Generate 100% factual, data-driven observations strictly derived from the actual dataset.
- Rules:
    * Only report facts supported by mathematics and actual values.
    * No speculative business claims.
    * No generative AI hallucinations.
    * Pure statistical and structural observations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def generate_factual_insights(df: pd.DataFrame, initial_df: pd.DataFrame = None) -> List[Dict[str, str]]:
    """
    Generate structured, factual insights from DataFrame.
    Returns a list of dicts with:
    - 'category': e.g., 'Dataset Structure', 'Data Quality', 'Distribution', 'Correlation'
    - 'icon': emoji or visual cue
    - 'headline': brief summary
    - 'detail': exact factual explanation with numbers
    """
    if df is None or df.empty:
        return [{
            "category": "Status",
            "icon": "ℹ️",
            "headline": "No Data Available",
            "detail": "The dataset is currently empty. Please load or upload a valid dataset."
        }]

    insights = []
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    # 1. Structure Insight
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    insights.append({
        "category": "Dataset Structure",
        "icon": "📊",
        "headline": f"Dataset contains {total_rows:,} records across {total_cols} columns",
        "detail": f"The dataset holds {total_cells:,} total data cells, consisting of {len(num_cols)} numerical column(s) and {len(cat_cols)} categorical/text column(s)."
    })

    # 2. Missing Value Facts
    missing_series = df.isna().sum()
    total_missing = int(missing_series.sum())
    if total_missing > 0:
        worst_col = missing_series.idxmax()
        worst_val = int(missing_series[worst_col])
        worst_pct = round(worst_val / total_rows * 100, 1)
        insights.append({
            "category": "Data Quality",
            "icon": "⚠️",
            "headline": f"{total_missing} missing value(s) detected across dataset",
            "detail": f"Column '{worst_col}' has the highest number of missing values with {worst_val} missing entries ({worst_pct}% of column rows)."
        })
    else:
        insights.append({
            "category": "Data Quality",
            "icon": "✅",
            "headline": "Zero missing values in active dataset",
            "detail": "All columns have complete data with 100% non-null record completeness."
        })

    # 3. Duplicate Records Fact
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / total_rows * 100, 1)
        insights.append({
            "category": "Data Quality",
            "icon": "🔄",
            "headline": f"{dup_count} duplicate record(s) detected ({dup_pct}%)",
            "detail": f"There are {dup_count} exact duplicate rows that share identical values across all columns."
        })
    else:
        insights.append({
            "category": "Data Quality",
            "icon": "✨",
            "headline": "No duplicate rows detected",
            "detail": "Every record in the current dataset is unique across all columns."
        })

    # 4. Numerical Extremes & Averages
    if num_cols:
        means = df[num_cols].mean()
        highest_avg_col = means.idxmax()
        lowest_avg_col = means.idxmin()

        insights.append({
            "category": "Numerical Averages",
            "icon": "📈",
            "headline": f"Highest average observed in '{highest_avg_col}' (Mean: {means[highest_avg_col]:.2f})",
            "detail": f"Among all numerical features, '{highest_avg_col}' has the highest mean value ({means[highest_avg_col]:.2f}), while '{lowest_avg_col}' exhibits the lowest mean ({means[lowest_avg_col]:.2f})."
        })

        # Single absolute max and min
        maxs = df[num_cols].max()
        mins = df[num_cols].min()
        abs_max_col = maxs.idxmax()
        abs_max_val = maxs[abs_max_col]
        abs_min_col = mins.idxmin()
        abs_min_val = mins[abs_min_col]

        insights.append({
            "category": "Extremes",
            "icon": "🎯",
            "headline": f"Peak value is {abs_max_val:.2f} in '{abs_max_col}'",
            "detail": f"The maximum observed value across numerical data is {abs_max_val:.2f} (in column '{abs_max_col}'), and the minimum observed value is {abs_min_val:.2f} (in column '{abs_min_col}')."
        })

        # Variation / Spread
        stds = df[num_cols].std()
        if not stds.isna().all():
            most_spread_col = stds.idxmax()
            insights.append({
                "category": "Data Spread",
                "icon": "📐",
                "headline": f"Highest dispersion found in '{most_spread_col}' (Std Dev: {stds[most_spread_col]:.2f})",
                "detail": f"Column '{most_spread_col}' exhibits the highest standard deviation ({stds[most_spread_col]:.2f}), indicating significant variation across records."
            })

    # 5. Correlation Facts
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        # Find strongest off-diagonal correlation
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append((c1, c2, val))

        if pairs:
            # Sort by absolute correlation
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pair = pairs[0]
            corr_val = top_pair[2]
            relationship = "strong positive" if corr_val >= 0.7 else "moderate positive" if corr_val > 0.3 else "strong negative" if corr_val <= -0.7 else "moderate negative" if corr_val < -0.3 else "weak"

            insights.append({
                "category": "Correlation Analysis",
                "icon": "🔗",
                "headline": f"Strongest relationship: '{top_pair[0]}' and '{top_pair[1]}' (r = {corr_val:.2f})",
                "detail": f"A {relationship} linear correlation of {corr_val:.2f} exists between '{top_pair[0]}' and '{top_pair[1]}'."
            })

    # 6. Categorical Dominance Facts
    if cat_cols:
        sample_cats = [c for c in cat_cols if df[c].nunique() > 1 and df[c].nunique() < total_rows]
        if sample_cats:
            cat_target = sample_cats[0]
            top_mode = df[cat_target].mode()
            if not top_mode.empty:
                val = top_mode.iloc[0]
                cnt = int((df[cat_target] == val).sum())
                pct = round(cnt / total_rows * 100, 1)
                insights.append({
                    "category": "Categorical Distribution",
                    "icon": "🏷️",
                    "headline": f"Dominant category in '{cat_target}': '{val}' ({pct}%)",
                    "detail": f"In column '{cat_target}', '{val}' appears {cnt} times, accounting for {pct}% of the records."
                })

    # 7. Cleaning History comparison if initial_df provided
    if initial_df is not None:
        init_rows = len(initial_df)
        init_missing = int(initial_df.isna().sum().sum())
        init_dups = int(initial_df.duplicated().sum())

        rows_cleaned = init_rows - total_rows
        missing_resolved = init_missing - total_missing
        dups_removed = init_dups - dup_count

        if rows_cleaned > 0 or missing_resolved > 0 or dups_removed > 0:
            insights.append({
                "category": "Cleaning Impact",
                "icon": "🧹",
                "headline": f"Cleaning improved dataset quality",
                "detail": f"Resolved {missing_resolved} missing value(s), removed {dups_removed} duplicate row(s), with a net row adjustment of {rows_cleaned} row(s)."
            })

    return insights
