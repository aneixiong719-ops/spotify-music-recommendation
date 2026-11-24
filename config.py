"""Shared paths and settings for the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"

# ---- raw inputs (see README for how each file is obtained) ----
LISTENS_RAW = RAW / "listenbrainz_listens.csv"          # ListenBrainz export
MBID_MAP = RAW / "mbid_spotify_mapping.json"            # MusicBrainz -> Spotify IDs (optional)
EMBEAT_RAW = RAW / "embeat_tracks.csv"                  # Embeat matched tracks + candidate sample
BATCH_FILE = ROOT / "batch.csv"                         # written by batch_sim.py
LISTENS_STREAM = RAW / "listens_from_batches.csv"       # accumulated batches

# ---- clean outputs ----
EVENTS_CLEAN = CLEAN / "listens_integrated_clean.csv"
TRACKS_CLEAN = CLEAN / "tracks_clean.csv"
USER_INDICATORS = CLEAN / "user_indicators.csv"
RECOMMENDATIONS = CLEAN / "recommendations.csv"
MODEL_METRICS = REPORTS / "model_metrics.json"
EXAM_REPORT = REPORTS / "data_exam_report.md"

AUDIO_FEATURES = ["danceability", "energy", "valence", "tempo",
                  "acousticness", "instrumentalness", "speechiness", "loudness"]
CORE_FEATURES = ["danceability", "energy", "valence", "tempo"]
K = 10                      # top-K for recommendations
N_MOOD_CLUSTERS = 5
RANDOM_STATE = 42

for d in (RAW, CLEAN, REPORTS):
    d.mkdir(parents=True, exist_ok=True)
