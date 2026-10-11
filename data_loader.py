"""
Data loading utility for tabular datasets.

Supports CSV, TSV, Excel (.xlsx, .xls), JSON, and Parquet formats.
Provides clear error messages for unsupported file formats.
"""

from pathlib import Path
import csv
import io
from typing import Union
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}


def detect_delimiter(sample: str, default: str = ",") -> str:
    """
    Sniff the delimiter from a text sample.
    Defaults to comma if sniffing fails.
    """
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        # Fallback heuristic: check if tabs appear more often than commas in the sample
        first_line = sample.splitlines()[0] if sample else ""
        if first_line.count("\t") > first_line.count(","):
            return "\t"
        return default


def load_table(source: Union[str, Path, io.BytesIO, io.StringIO, object]) -> pd.DataFrame:
    """
    Load a tabular dataset into a pandas DataFrame.

    Parameters:
        source: File path (str or Path) or file-like buffer (e.g. Streamlit UploadedFile).

    Returns:
        pd.DataFrame containing the loaded dataset.

    Raises:
        ValueError: If the file format is unsupported or the file cannot be parsed.
        FileNotFoundError: If the file path does not exist.
    """
    # Determine the file name and extension
    file_name = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        file_name = path.name
    elif hasattr(source, "name") and isinstance(source.name, str):
        file_name = source.name

    if not file_name:
        raise ValueError("Could not determine file type. Please provide a valid file path or named buffer.")

    ext = Path(file_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: {supported}"
        )

    try:
        if ext in {".csv", ".tsv"}:
            # Handle text delimiter sniffing
            if isinstance(source, (str, Path)):
                with open(source, "r", encoding="utf-8", errors="replace") as f:
                    sample = f.read(4096)
                delim = "\t" if ext == ".tsv" else detect_delimiter(sample, default=",")
                return pd.read_csv(source, sep=delim)
            else:
                # File-like object (e.g. UploadedFile)
                if hasattr(source, "seek"):
                    source.seek(0)
                # Read a snippet for sniffing
                raw_bytes = source.read(4096)
                if hasattr(source, "seek"):
                    source.seek(0)

                sample_text = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
                delim = "\t" if ext == ".tsv" else detect_delimiter(sample_text, default=",")
                return pd.read_csv(source, sep=delim)

        elif ext in {".xlsx", ".xls"}:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_excel(source, sheet_name=0)

        elif ext == ".json":
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_json(source)

        elif ext == ".parquet":
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_parquet(source)

        else:
            raise ValueError(f"Unsupported file format '{ext}'.")

    except Exception as e:
        if isinstance(e, ValueError) and "Unsupported file format" in str(e):
            raise
        raise ValueError(f"Failed to read file '{file_name}': {e}") from e
