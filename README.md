# Social Graphs and Artist Lyrics

This project builds a collaboration network of pop artists and explores whether artists in the same collaboration community use more similar lyrics. It contains two related workflows: network construction and lyric embedding analysis.

## Project structure

```text
social_graphs_and_interactions/
├── scripts_network/
│   ├── 1–7 pipeline scripts
│   ├── analysis/
│   └── visualisations/
├── scripts_embeddings/
│   ├── 1–6 pipeline scripts
│   └── analysis and visualisation scripts
```

The network workflow uses `data/data_2005_2025_network`. The lyrics and embedding workflow uses `data/data_2005_2025_artist_embeddings`. Scripts that compare embeddings with network communities also read `data/data_2005_2025_network/network/louvain_communities.csv`.

All files under `data/` are kept locally and ignored by Git. They are not included in a fresh clone, so the datasets must be obtained separately before running the workflows. Do not commit API credentials; `.env` is also ignored.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- A ListenBrainz token for fetching popular recordings
- Internet access for the MusicBrainz, ListenBrainz, and LRCLIB fetch steps

Install the project dependencies from the repository root:

```bash
uv sync
```

For step 1 of the embeddings workflow, create a local `.env` file in the repository root:

```dotenv
LISTENBRAINZ_TOKEN=your_token
```

## Workflow A: collaboration network

Run the scripts from the repository root and in order. The fetch steps call external services and can take a long time.

| Step | Script | Purpose and main output |
|---|---|---|
| 1 | `scripts_network/1_find_pop_artists.py` | Finds artists with at least six unique pop-tagged recordings in the configured period. Writes `pop_artists.csv`. |
| 2 | `scripts_network/2_fetch_data.py` | Fetches recordings for the selected artists, applies the year and alternate-version filters, and writes `artists.csv` and `recordings.csv`. |
| 3.0 | `scripts_network/3_0_preprocess_data.py` | Consolidates recordings with the same normalized title and artist set. Writes `deduplication/recordings_for_dedup.csv` and supporting reports. |
| 3.1 | `scripts_network/3_1_deduplicate_recordings.py` | Excludes configured cases and groups likely versions of the same song. Writes `unique_songs.csv` and a recording-to-song mapping. |
| 3.2 | `scripts_network/3_2_deduplicate_rapidfuzz.py` | Generates candidate duplicate pairs for manual review, then applies the decisions in `potential_duplicates_review.csv`. Writes `unique_songs_reviewed.csv`. |
| 4 | `scripts_network/4_build_network.py` | Builds `network/nodes.csv`, `network/edges.csv`, an edge-to-song audit, and network metadata from the reviewed songs. |
| 5 | `scripts_network/5_export_louvain_communities_org_net.py` | Finds NetworkX Louvain communities in the original network at resolution 1.2 and seed 42. Exports a GEXF file for Gephi and writes `louvain_communities.csv`. |
| 6 | `scripts_network/6_disparicy_filter.py` | Applies the disparity filter at several alpha thresholds. Writes the edge tables under `network/backbone/`. |
| 7 | `scripts_network/7_export_backbone_with_org_communities.py` | Exports the alpha 0.42 backbone to Gephi, retaining each artist’s community from the original network. |

The main network files are under `data/data_2005_2025_network/network/`. Step 5 analyses the full network; step 7 exports the filtered backbone with the original-network community labels.

Example commands:

```bash
uv run python scripts_network/1_find_pop_artists.py
uv run python scripts_network/2_fetch_data.py
uv run python scripts_network/3_0_preprocess_data.py
uv run python scripts_network/3_1_deduplicate_recordings.py
uv run python scripts_network/3_2_deduplicate_rapidfuzz.py
uv run python scripts_network/4_build_network.py
uv run python scripts_network/5_export_louvain_communities_org_net.py
uv run python scripts_network/6_disparicy_filter.py
uv run python scripts_network/7_export_backbone_with_org_communities.py
```

Step 3.2 requires manual review. When the review file already exists, choose the option to apply the saved decisions; regenerating it replaces the candidate list.

## Workflow B: lyrics and embeddings

This workflow uses the artist list from `data/data_2005_2025_network/artists.csv` and stores its data under `data/data_2005_2025_artist_embeddings/`.

| Step | Script | Purpose and main output |
|---|---|---|
| 1 | `scripts_embeddings/1_get_top_recordings.py` | Gets popular recordings for each artist from ListenBrainz, checks release years with MusicBrainz, and saves a resumable top-recordings table. |
| 2 | `scripts_embeddings/2_get_lyrics.py` | Searches LRCLIB for lyrics and saves verified matches, missing matches, and instrumentals. |
| 3 | `scripts_embeddings/3_generate_embeddings.py` | Uses `paraphrase-multilingual-mpnet-base-v2` to create song and artist embedding arrays and their metadata. |
| 4 | `scripts_embeddings/4_inspect_artist_embeddings.py` | Reports similar artist pairs and nearest neighbors, with their original-network communities. |
| 5 | `scripts_embeddings/5_inspect_song_embeddings.py` | Reports similar song pairs and nearest songs for each song. |
| 6 | `scripts_embeddings/6_analyse_similarity.py` | Compares artist-embedding similarity within and between original-network communities and runs a one-sided permutation test (10,000 permutations). |

The numbered workflow writes lyrics to `hentet_lyrics_2/`, embeddings to `embeddings/v2/`, and semantic outputs to `semantic_analysis/` or `semantic_analysis_2/` inside the artist-embeddings data directory.

Run from the repository root:

```bash
uv run python scripts_embeddings/1_get_top_recordings.py
uv run python scripts_embeddings/2_get_lyrics.py
uv run python scripts_embeddings/3_generate_embeddings.py
uv run python scripts_embeddings/4_inspect_artist_embeddings.py
uv run python scripts_embeddings/5_inspect_song_embeddings.py
uv run python scripts_embeddings/6_analyse_similarity.py
```

The fetch scripts save progress so interrupted retrievals can resume. Embedding generation and later analyses can be rerun from the saved local data without repeating the lyrics API calls.

## Additional analysis and quality checks

- `scripts_network/analysis/analyze_network_properties.py` calculates descriptive statistics and node-level metrics for the original network, and compares selected properties with the alpha 0.42 backbone. It writes tables and plots to `data/data_2005_2025_network/network/network_properties/`. Run it from the repository root with `uv run python scripts_network/analysis/analyze_network_properties.py`.
- `scripts_embeddings/audit_lyrics.py` checks the current lyrics file for normalized duplicate texts, title/artist mismatches, and a small random sample for manual review. Reports go to `quality_audit/`.
- `scripts_embeddings/robustness_test.py` compares the community-similarity result for all artists with the result after requiring at least three songs per artist.
- `scripts_embeddings/visualize_communities_on_embeddings.py` creates a UMAP view of artist embeddings colored by original-network community.
- `scripts_embeddings/visualize_similarities.py` plots pairwise similarity distributions from step 6.
- `scripts_embeddings/visualise_permutaions.py` plots the permutation-test null distribution from step 6.
- `scripts_network/analysis/analyze_resolution_sensitivity.py` compares NetworkX partitions at resolutions 0.5, 0.8, 1.0, 1.2, and 1.5. It compares their standard modularity at gamma 1 and their pairwise NMI.
- `scripts_network/analysis/analyze_louvain_stability.py` measures NetworkX partition stability over 50 seeds at the selected resolution (currently 1.2).
- Resolution 1.2 is selected because its partition had the highest standard modularity (gamma=1) among the tested resolutions.
- `scripts_network/analysis/compare_louvain_implementations.py` is a diagnostic comparison with the earlier `python-louvain` implementation; it is not part of the final analysis pipeline.
- `scripts_network/visualisations/` contains supporting network plots; these are not extra numbered pipeline steps.

## Data and reproducibility notes

- Network edges are weighted by the number of unique reviewed songs shared by each artist pair.
- The backbone is a filtered version of the original network. It does not replace `network/edges.csv`.
- Embeddings depend on the saved lyrics, the selected model, and the row order in the matching metadata CSVs. Keep each `.npy` file together with its corresponding metadata file.
- Fetching data again may update existing CSVs and progress files. Review the configured inputs and outputs before rerunning a fetch or deduplication step.
