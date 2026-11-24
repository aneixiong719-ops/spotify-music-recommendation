# From plays to personalization — music recommendation with listening behavior × audio features

Data Science Toolbox group project (Rennes School of Business).

## Data sources
| Source | What it gives us | Link |
|---|---|---|
| **ListenBrainz** | Public listening histories (who played what, when), Jan 2025 – Nov 2026 | https://listenbrainz.org/data/ |
| **Embeat** | Spotify tracks with audio features: danceability, energy, valence, tempo, acousticness… | https://huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks |
| MusicBrainz | Recovers Spotify IDs that ListenBrainz does not provide directly | https://musicbrainz.org |

The data was collected in 2026.

## Questions
1. **What does each listener's taste look like, and how varied is it?**
   Indicators: taste profile (mean audio features), taste spread, repeat share, artist diversity, dominant mood cluster.
2. **Can audio features recommend songs a listener will actually play next?**
   Indicators: content-based recommender vs. popularity baseline, evaluated with Precision@10, Recall@10, NDCG@10 and catalog coverage on a chronological hold-out.

## Repository structure
```
data/raw/          inputs (ListenBrainz listens, MBID→Spotify mapping, Embeat tracks)
data/clean/        cleaned + integrated outputs (generated)
reports/           data examination report, model metrics, pipeline report (generated)
config.py          paths and settings
data_exam.py       data examination + cleaning + ListenBrainz ↔ Embeat join
ml_models.py       KMeans mood clusters + content-based recommender + evaluation
indicators.py      user-level indicators for Q1 / Q2
pipeline.py        ingest new batch → clean → ML → indicators → report
batch_sim.py       simulates new listens arriving in batches (pushes batch.csv)
app.py             Dash dashboard (data story)
.github/workflows/pipeline.yml   GitHub Actions automation
```

## Input files expected in `data/raw/`
| File | Required columns |
|---|---|
| `listenbrainz_listens.csv` | `user_id, listened_at, track_name, artist_name` (+ optional `recording_mbid`, `spotify_url`) |
| `listenbrainz_new_listens.csv` | same columns — the most recent listens, held back for the batch simulation |
| `mbid_spotify_mapping.json` | list of `{recording_mbid, spotify_track_ids: [...]}` (optional) |
| `embeat_tracks.csv` | `track_id, track_name, artist_name` + audio features. Contains the matched tracks **plus a random sample of Embeat tracks** used as recommendation candidates (the full 9.5 GB dump is not stored). |

## How to run
```bash
pip install -r requirements.txt
python pipeline.py      # or step by step: data_exam.py → ml_models.py → indicators.py
python app.py           # open http://127.0.0.1:8050
```

## Automation
`batch_sim.py` releases the held-back listens in chronological order, writes `batch.csv` and pushes it.
The push triggers **GitHub Actions** (`.github/workflows/pipeline.yml`), which re-runs the whole pipeline and commits the updated clean data, indicators, model metrics and `reports/report.md` back to the repo.

```bash
python batch_sim.py data/raw/listenbrainz_new_listens.csv --batch-size 50 --n-batches 3
```

Design choices:
- Triggered **on push of new data** (not on a schedule): processing only happens when something changed.
- `concurrency` keeps batches in order, so two batches never overwrite each other.
- Results are committed with `[skip ci]` to avoid an infinite loop of runs.
- The report is also written to the Actions run summary for a quick look.

## Key cleaning decisions
- Listens: timestamps converted from Unix seconds, exact duplicates removed, rows missing user/time/track dropped.
- Join: a listen is linked to Embeat only through **one** unambiguous Spotify ID (direct, or a single MusicBrainz candidate); ambiguous mappings are kept but not used for modeling.
- Metadata check: title/artist similarity with Unicode-aware normalization (an earlier Latin-only rule wrongly rejected Japanese artist names).
- All original listens are kept; a `model_candidate` flag marks the high-agreement ones.

## Team workflow
Each member works on a feature branch (`data-cleaning`, `ml-model`, `dashboard`) and merges into `main` through a pull request.
