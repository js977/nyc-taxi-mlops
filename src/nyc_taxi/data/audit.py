
import pandas as pd


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return a data quality summary for a DataFrame."""

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    return report
