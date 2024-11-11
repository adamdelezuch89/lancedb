import pandas as pd
from typing import Dict
import ast
import pyarrow as pa
import warnings


def get_arrow_schema(df: pd.DataFrame) -> pa.Schema:
    """
    Detect column types and map them to PyArrow schema.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pa.Schema: PyArrow schema object
    """
    type_mapping = {}

    for column in df.columns:
        non_null_values = df[column][df[column].notna()].tolist()

        if len(non_null_values) == 0:
            type_mapping[column] = pa.string()
            continue

        # Check if it's a float column first
        if df[column].dtype in ["float64", "float32"]:
            type_mapping[column] = pa.float64()
            continue

        # For other numeric types, check more carefully
        try:
            values = [float(str(x).strip()) for x in non_null_values]

            # If any value has a decimal part, it's a float
            if any(not x.is_integer() for x in values):
                type_mapping[column] = pa.float64()
                continue

            # If all values are integers, check range
            values = [int(x) for x in values]
            if all(-(2**31) <= x <= 2**31 - 1 for x in values):
                type_mapping[column] = pa.int32()
            else:
                type_mapping[column] = pa.int64()
            continue

        except (ValueError, TypeError):
            pass

        # Check boolean
        unique_values = set(str(x).lower().strip() for x in non_null_values)
        if unique_values.issubset({"true", "false", "1", "0", "yes", "no", "y", "n"}):
            type_mapping[column] = pa.bool_()
            continue

        # Try datetime
        try:
            # Sample a few values to check if they look like dates
            sample = non_null_values[:5]

            # Check for date-like characters
            date_indicators = ["-", "/", ":", "T", "Z"]
            if any(
                any(indicator in str(x) for indicator in date_indicators)
                for x in sample
            ):
                # Try common date formats
                date_formats = [
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]

                for format in date_formats:
                    try:
                        pd.to_datetime(non_null_values, format=format)
                        type_mapping[column] = pa.timestamp("ns")
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matches, try the flexible parser with warning suppressed
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            pd.to_datetime(non_null_values)
                            type_mapping[column] = pa.timestamp("ns")
                        except (ValueError, TypeError):
                            type_mapping[column] = pa.string()
                continue
        except (ValueError, TypeError):
            pass

        # Check list
        if all(
            str(x).strip().startswith("[") and str(x).strip().endswith("]")
            for x in non_null_values
        ):
            try:
                sample_lists = [
                    ast.literal_eval(str(x).strip()) for x in non_null_values[:100]
                ]
                if not sample_lists:
                    type_mapping[column] = pa.list_(pa.string())
                    continue

                for sample in sample_lists:
                    if sample:
                        first_elem = sample[0]
                        if isinstance(first_elem, bool):
                            type_mapping[column] = pa.list_(pa.bool_())
                        elif isinstance(first_elem, int):
                            type_mapping[column] = pa.list_(pa.int64())
                        elif isinstance(first_elem, float):
                            type_mapping[column] = pa.list_(pa.float64())
                        else:
                            type_mapping[column] = pa.list_(pa.string())
                        break
                else:
                    type_mapping[column] = pa.list_(pa.string())
                continue
            except (ValueError, SyntaxError):
                type_mapping[column] = pa.string()
                continue

        # Default to string
        type_mapping[column] = pa.string()

    # Convert dictionary to schema at the end
    return pa.schema(
        [
            pa.field(
                name,
                dtype,
                nullable=df[name]
                .isnull()
                .any(),  # Make field nullable if there are any null values
            )
            for name, dtype in type_mapping.items()
        ]
    )
