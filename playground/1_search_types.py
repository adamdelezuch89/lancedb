import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import Vector, LanceModel
import dotenv


dotenv.load_dotenv()

db = lancedb.connect(".lancedb/query_types")
df = pd.read_csv(".data/data_qa.csv")


table = db.open_table("qa")

queries = df["query"].tolist()
print("dataset:")
print(df.head())
print("vector:")
print(table.search(queries[0], query_type="vector").limit(5).to_pandas())
print("fts:")
table.create_fts_index("context", replace=True)
print(table.search(queries[0], query_type="fts").limit(5).to_pandas())
print("hybrid:")
print(table.search(queries[0], query_type="hybrid").limit(5).to_pandas())
