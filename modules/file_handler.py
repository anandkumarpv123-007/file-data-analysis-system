"""
Module 1: Data Input & File Handling
====================================
Responsibilities:
- Read CSV and supported tabular TXT files.
- Automatically detect delimiters (comma, tab, semicolon, pipe).
- Validate file contents (empty file, invalid syntax, unparseable data).
- Generate dataset preview and metadata (rows, columns, dtypes).
"""

import io
import csv
import pandas as pd
from typing import Tuple, Dict, Any, Optional


def detect_delimiter(file_sample: str) -> str:
    """
    Detect the delimiter of a tabular text sample using csv.Sniffer
    with fallbacks for comma, tab, semicolon, and pipe.
    """
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(file_sample, delimiters=[',', '\t', ';', '|'])
        return dialect.delimiter
    except Exception:
        # Fallback check based on common delimiter frequency
        counts = {
            ',': file_sample.count(','),
            '\t': file_sample.count('\t'),
            ';': file_sample.count(';'),
            '|': file_sample.count('|'),
        }
        best_delimiter = max(counts, key=counts.get)
        return best_delimiter if counts[best_delimiter] > 0 else ','


def load_dataset(file_or_path: Any, file_name: str = "") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read an uploaded file or file path into a pandas DataFrame.
    
    Handles:
    - Empty files
    - CSV files
    - Tabular TXT files (tab, semicolon, pipe delimited)
    - Encoding issues (utf-8, latin1 fallback)
    
    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    if file_or_path is None:
        return None, "No file provided."

    try:
        # Read raw bytes or string
        if hasattr(file_or_path, 'read'):
            # Streamlit UploadedFile or file-like object
            raw_bytes = file_or_path.read()
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)  # Reset pointer for re-reading if needed
        elif isinstance(file_or_path, str):
            with open(file_or_path, 'rb') as f:
                raw_bytes = f.read()
        elif isinstance(file_or_path, bytes):
            raw_bytes = file_or_path
        else:
            return None, "Unsupported file input type."

        # Check for empty file
        if not raw_bytes or len(raw_bytes.strip()) == 0:
            return None, "The uploaded file is empty."

        # Decode content with UTF-8 or Latin-1 fallback
        try:
            content_str = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content_str = raw_bytes.decode('latin-1')
            except Exception as e:
                return None, f"Failed to decode file encoding: {str(e)}"

        # Verify content has more than empty whitespace
        if not content_str.strip():
            return None, "The uploaded file contains no data."

        # Sample for delimiter detection
        sample_lines = content_str.splitlines()[:20]
        sample_text = "\n".join(sample_lines)
        delimiter = detect_delimiter(sample_text)

        # Parse with pandas
        try:
            df = pd.read_csv(
                io.StringIO(content_str),
                sep=delimiter,
                engine='python',
                on_bad_lines='skip'
            )
        except Exception:
            # Fallback to comma if detected delimiter failed
            df = pd.read_csv(
                io.StringIO(content_str),
                engine='python',
                on_bad_lines='skip'
            )

        # Validate resulting DataFrame
        if df is None or df.empty:
            return None, "Unable to read this file as a valid tabular dataset. (No rows found)"

        if len(df.columns) == 0:
            return None, "The file does not contain valid columns."

        # Clean column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        return df, None

    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty."
    except pd.errors.ParserError as pe:
        return None, f"Unable to read this file as a valid tabular dataset: {str(pe)}"
    except Exception as ex:
        return None, f"Unable to read this file as a valid tabular dataset: {str(ex)}"


def get_dataset_metadata(df: pd.DataFrame, file_name: str = "Dataset") -> Dict[str, Any]:
    """
    Extract basic metadata from the loaded DataFrame.
    """
    if df is None:
        return {}

    return {
        "file_name": file_name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_bytes": int(df.memory_usage(deep=True).sum())
    }


def get_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the first n rows as preview.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(n)
