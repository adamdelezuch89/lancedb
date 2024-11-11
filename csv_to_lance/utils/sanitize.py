import pandas as pd
import pyarrow as pa
import logging

logger = logging.getLogger(__name__)


def sanitize_data(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """
    Sanitize DataFrame to match the expected schema dynamically
    Args:
        df: pandas DataFrame
        schema: pyarrow Schema
    Returns:
        Sanitized DataFrame
    """
    import ast

    df = df.copy()

    for field in schema:
        col_name = field.name
        if col_name not in df.columns:
            continue

        # Handle list types
        if str(field.type).startswith("list"):
            df[col_name] = df[col_name].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else []
            )

        # Handle other types
        else:
            try:
                df[col_name] = df[col_name].astype(field.type.to_pandas_dtype())
            except Exception as e:
                logger.warning(f"Could not convert {col_name} to {field.type}: {e}")

    return df
