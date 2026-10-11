# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Corpus2GeoRAG turns a text corpus (Wikipedia articles) into a geo-enriched knowledge graph for RAG. There is no package, build system, test suite, or linter. The whole pipeline lives in one Colab-oriented notebook plus example data files.

## Pipeline (`Graph_n_Geocoding/Graph_and_Geocoding.ipynb`)

The notebook is written to run top-to-bottom in Google Colab (it uses `!pip install` and `%%bash`). Stages:

1. **Setup**: `pip install pyvis`, then `git clone https://github.com/EtzionR/Corpus2GeoRAG.git` so it can read `Corpus2GeoRAG/examples/entities_table.csv`. Locally, point `pd.read_csv` at `examples/entities_table.csv` instead.
2. **Read NER output**: the CSV is the result of a token-classification model run over the corpus, one row per entity.
3. **Geocode**: unique `LOC` entities go through geopy `Nominatim`, wrapped in `RateLimiter` at a 1-second minimum delay. Latitude, longitude and address are merged back into the DataFrame by `word`. A folium map (Esri tiles, `MarkerCluster`) visualizes the results.
4. **Build graph**: `MISC` entities are dropped. A `networkx.DiGraph` is built with the entity (`word`) as the node id. `LOC` nodes carry `latitude`, `longitude` and `address`. Edges join every pair of entities that share the same `(Source, Index)` text snippet. Each edge has `relation="CO_OCCURS_WITH"`, `evidence` (the snippet) and `source_doc`. The relation is a placeholder meant to be replaced by LLM-extracted predicates.
5. **Export and visualize**: `nx.node_link_data(G)` is written to `knowledge_graph.json` with `ensure_ascii=False`. A PyVis network is rendered inline, and the top-10 nodes by degree are printed.

## Data files

- `examples/DATA.json`: corpus as `{article_title: full_text}` (about 897 articles).
- `examples/entities_table.csv`: NER output with columns `entity_group` (PER, LOC, ORG, MISC), `score`, `word`, `start`, `end`, `Source` (article title), `Index` (snippet index within the article), `Text` (the snippet). `(Source, Index)` identifies a context window.
- `Graph_n_Geocoding/knowledge_graph.json`: committed output of the notebook. Its edge list uses the key `"edges"` (the networkx 3.4+ default for `node_link_data`), not `"links"`.

## Notes

- Nominatim enforces usage limits. Keep the `RateLimiter` and a descriptive `user_agent`. Geocoding all LOC entities is slow.
- The notebook writes `knowledge_graph.json` to the working directory. In Colab that is `/content`, so it must be copied into `Graph_n_Geocoding/` to update the committed copy.
- The notebook's `%%bash` cell runs `rm -rf LM4GeoAI`, a stale directory name that does not match the repo being cloned.
- When editing the notebook while it is open in VS Code, follow the user-level instructions: use the `notebook-mcp` tools, not direct file edits.
