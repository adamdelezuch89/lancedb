import lancedb
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import get_registry
import pyarrow as pa
from lancedb.embeddings.base import EmbeddingFunctionConfig


db = lancedb.connect(".lancedb/test")
if "documents" in db.table_names():
    db.drop_table("documents")

# Ways to get embeddings model
# 1. OpenAI
# embeddings = get_registry().get("openai").create()

# 2. Sentence-Transformers
embeddings = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")


# Ways to create schema
# 1. Using LanceModel
# class Documents(LanceModel):
#     vector: Vector(embeddings.ndims()) = embeddings.VectorField()
#     text: str = embeddings.SourceField()
# table = db.create_table("documents", schema=Documents)

# 2. Using pyarrow schema
schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), embeddings.ndims())),
        pa.field("text", pa.string()),
    ]
)

embedding_functions = EmbeddingFunctionConfig(
    function=embeddings,
    source_column="text",
    vector_column="vector",
)

table = db.create_table(
    "documents",
    schema=schema,
    mode="overwrite",
    embedding_functions=[embedding_functions],
)


# Insert data
data = [
    {"text": "rebel spaceships striking from a hidden base"},
    {"text": "have won their first victory against the evil Galactic Empire"},
    {"text": "during the battle rebel spies managed to steal secret plans"},
    {"text": "hi man"},
]

table.add(data)

query = "greetings"
actual = table.search(query).limit(1).to_list()
print(actual)
