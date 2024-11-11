python csv_to_lance/main.py \
    --data movies.csv \
    --columns title extract \
    --db movies \
    --table movies

python display_lance_table/main.py \
  --db movies \
  --table movies

  python csv_to_lance/main.py \
    --data data_qa.csv \
    --columns context \
    --db query_types \
    --table qa

python playground/4_search_movie_by_desc.py \
    --description "movie about man in rabbit costume which say about end of world" \
    --table movies \
    --db movies \
    --limit 5