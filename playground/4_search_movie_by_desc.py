import os
import argparse
import lancedb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def search_movies_by_description(db_path, table_name, description, limit=5):

    # Connect to LanceDB
    db = lancedb.connect(db_path)

    # Get the table
    table = db[table_name]

    # table.create_index(
    #     num_partitions=16, num_sub_vectors=48, vector_column_name="extract_vector"
    # )
    # table.create_fts_index("extract", use_tantivy=False)
    # Search using vector similarity
    results = (
        table.search(
            description,
            vector_column_name="extract_vector",
            query_type="hybrid",
        )
        .limit(limit)
        .to_pandas()
    )
    print(results)
    return results


def main():
    # Load environment variables
    load_dotenv()

    # Get LanceDB directory from environment variable
    lance_db_dir = os.getenv("LANCE_DB_DIR")
    if not lance_db_dir:
        raise ValueError("LANCE_DB_DIR environment variable not set")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search movies by description")
    parser.add_argument(
        "--description", "-d", required=True, help="Description to search for"
    )
    parser.add_argument("--table", "-t", default="movies", help="Table name in LanceDB")
    parser.add_argument("--db", "-b", default="movies", help="Database name in LanceDB")
    parser.add_argument(
        "--limit", "-l", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    try:
        lance_db_path = os.path.join(lance_db_dir, args.db)
        # Search for movies
        results = search_movies_by_description(
            lance_db_path, args.table, args.description, args.limit
        )

        # Print results
        print("\nSearch Results:")
        print("-" * 50)
        for _, row in results.iterrows():
            print(f"Title: {row['title']}")
            print(f"Description: {row['extract']}")
            print(f"Similarity Score: {row['_relevance_score']:.4f}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
