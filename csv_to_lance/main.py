import argparse
import logging
import pandas as pd
import lancedb
import os
from dotenv import load_dotenv
import pyarrow as pa
from lancedb.embeddings import get_registry
from sentence_transformers import SentenceTransformer
import numpy as np
from utils import get_arrow_schema, sanitize_data
from tqdm import tqdm
from lancedb.embeddings.base import EmbeddingFunctionConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    # Load environment variables
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", ".data")
    lance_db_dir = os.getenv("LANCE_DB_DIR", ".lancedb")

    # Construct full paths
    full_data_path = os.path.join(data_dir, args.data)
    full_db_path = os.path.join(lance_db_dir, args.db)

    logger.info(f"Loading data from {full_data_path}")
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Using database: {full_db_path}")

    df = pd.read_csv(full_data_path)
    df = df.dropna(subset=args.columns)

    # Connect to LanceDB and create/update table
    logger.info(f"Connecting to LanceDB at {full_db_path}")
    db = lancedb.connect(full_db_path)

    # Set up embeddings model
    embeddings = (
        get_registry().get("sentence-transformers").create(name=args.model_name)
    )

    schema = get_arrow_schema(df)
    fields = list(schema)  # Convert schema to list of fields
    df = sanitize_data(df, schema)

    # Add vector fields for each column to be embedded
    for column in args.columns:
        vector_field = pa.field(
            f"{column}_vector", pa.list_(pa.float32(), embeddings.ndims())
        )
        fields.append(vector_field)

    schema = pa.schema(fields)

    embedding_functions = [
        EmbeddingFunctionConfig(
            function=embeddings,
            source_column=column,
            vector_column=f"{column}_vector",
        )
        for column in args.columns
    ]

    # Create or overwrite table
    table = db.create_table(
        args.table,
        schema=schema,
        mode="overwrite",
        embedding_functions=embedding_functions,
    )

    logger.info(f"Adding {len(df)} rows to table {args.table}")
    table.add(df, on_bad_vectors="skip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize data and store in LanceDB")
    parser.add_argument("--data", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Columns to vectorize (space-separated)",
    )
    parser.add_argument("--db", required=True, help="Name of the LanceDB database")
    parser.add_argument("--table", required=True, help="Name of the table in LanceDB")
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model (e.g., 'all-MiniLM-L6-v2' or 'openai'). Default: all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing embeddings and adding to database. Default: 1000",
    )

    args = parser.parse_args()
    main(args)
