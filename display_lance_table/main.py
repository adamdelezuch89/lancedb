import argparse
import lancedb
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Display Lance table contents")
    parser.add_argument(
        "--db", type=str, required=True, help="Name of the Lance database"
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Name of the Lance table to display",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_dir = Path(os.getenv("LANCE_DB_DIR", ".lancedb"))
    db_name = args.db
    table_name = args.table

    db_path = db_dir / db_name
    logger.info(f"Loading Lance database from: {db_path}")
    logger.info(f"Accessing table: {table_name}")

    try:
        # Load Lance dataset
        db = lancedb.connect(db_path)
        table = db[table_name]

        # Display schema
        logger.info("\n\nTable Schema:")
        print(table.schema)

        # Convert to pandas DataFrame
        df = table.to_pandas()

        logger.info("\n\nDataFrame Info:")
        df.info()
        

    except Exception as e:
        logger.error(f"Error processing Lance table: {str(e)}")
        raise


if __name__ == "__main__":
    main()
