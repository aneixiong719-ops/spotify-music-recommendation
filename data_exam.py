"""Data examination and cleaning.

1. Examine both sources: columns, types, min/max, nulls, duplicates, misspellings.
2. Clean ListenBrainz listening events and Embeat track features.
3. Link each listen to one Embeat track (direct Spotify ID, else a unique
   MusicBrainz -> Spotify mapping) and flag metadata agreement.
4. Save the cleaned files to data/clean/ and a written report to reports/.

Run:  python data_exam.py
"""
import json
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import config as C

# ----------------------------------------------------------------- helpers
SPOTIFY_PATTERN = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/(?:intl-[a-z]{2}/)?track/)"
    r"([A-Za-z0-9]{22})(?=$|[/?#])", re.IGNORECASE)
MBID_PATTERN = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}")


def spotify_track_id(value):
    """Extract a 22-char Spotify track ID from an ID, URI or URL."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9]{22}", value):
        return value
    m = SPOTIFY_PATTERN.search(value)
    return m.group(1) if m else None


def mbid_key(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if MBID_PATTERN.fullmatch(v) else None


def normalize_text(value):
    """Case-fold, strip accents, keep letters of ALL scripts (e.g. Japanese).

    An earlier version used a Latin-only regex, which wiped Japanese artist
    names and produced false mismatches -- this keeps any Unicode letter.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def similarity(a, b):
    a, b = normalize_text(a), normalize_text(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def metadata_status(track_sim, artist_sim):
    if track_sim >= 0.90 and artist_sim >= 0.80:
        return "High Agreement"
    if track_sim >= 0.65 and artist_sim >= 0.60:
        return "Needs Review"
    return "Possible Mismatch"


# ------------------------------------------------------------- examination
def profile(df, name):
    """Return a markdown block describing a dataframe."""
    lines = [f"## {name}", f"- Rows: {len(df):,} | Columns: {df.shape[1]}",
             f"- Fully duplicated rows: {int(df.duplicated().sum())}", "",
             "| column | dtype | nulls | null % | unique | min | max |",
             "|---|---|---|---|---|---|---|"]
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            mn, mx = s.min(), s.max()
            mn, mx = (f"{mn:.3g}", f"{mx:.3g}") if pd.notna(mn) else ("", "")
        else:
            mn = mx = ""
        lines.append(f"| {col} | {s.dtype} | {int(s.isna().sum())} | "
                     f"{100 * s.isna().mean():.1f} | {s.nunique(dropna=True)} | {mn} | {mx} |")
    return "\n".join(lines) + "\n"


def spelling_variants(series, label):
    """Names that differ only by case/accents/punctuation = likely misspellings."""
    s = series.dropna().astype(str)
    groups = s.groupby(s.map(normalize_text)).unique()
    variants = groups[groups.map(len) > 1]
    out = [f"### Spelling variants in `{label}`: {len(variants)} groups"]
    for _, vals in variants.head(10).items():
        out.append("- " + " / ".join(map(str, vals)))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- loading
def load_listens():
    frames = [pd.read_csv(C.LISTENS_RAW)]
    if C.LISTENS_STREAM.exists():                 # batches added by the pipeline
        frames.append(pd.read_csv(C.LISTENS_STREAM))
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    required = {"user_id", "listened_at", "track_name", "artist_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ListenBrainz file is missing columns: {missing}")
    return df


def load_mapping():
    if not C.MBID_MAP.exists():
        return {}
    maps = pd.DataFrame(json.loads(C.MBID_MAP.read_text(encoding="utf-8")))
    maps = maps[["recording_mbid", "spotify_track_ids"]].explode("spotify_track_ids")
    maps["mbid"] = maps["recording_mbid"].map(mbid_key)
    maps["sid"] = maps["spotify_track_ids"].map(spotify_track_id)
    maps = maps.dropna(subset=["mbid", "sid"])
    return maps.groupby("mbid")["sid"].agg(lambda v: sorted(set(v))).to_dict()


# --------------------------------------------------------------- cleaning
def clean_listens(df):
    df = df.copy()
    # timestamps: ListenBrainz gives unix seconds; accept ISO strings too
    ts = pd.to_numeric(df["listened_at"], errors="coerce")
    df["listened_at"] = np.where(ts.notna(),
                                 pd.to_datetime(ts, unit="s", utc=True, errors="coerce"),
                                 pd.to_datetime(df["listened_at"], utc=True, errors="coerce"))
    df["listened_at"] = pd.to_datetime(df["listened_at"], utc=True)
    for col in ("user_id", "track_name", "artist_name"):
        df[col] = df[col].astype("string").str.strip()
    before = len(df)
    df = df.dropna(subset=["user_id", "listened_at", "track_name"])
    df = df.drop_duplicates(subset=["user_id", "listened_at", "track_name", "artist_name"])
    df = df.sort_values(["user_id", "listened_at"]).reset_index(drop=True)
    df.insert(0, "event_id", np.arange(1, len(df) + 1))
    df["mbid"] = df.get("recording_mbid", pd.Series(index=df.index, dtype=object)).map(mbid_key)
    id_col = next((c for c in ("spotify_track_id", "spotify_id", "spotify_url") if c in df), None)
    df["direct_spotify_id"] = df[id_col].map(spotify_track_id) if id_col else None
    return df, before - len(df)


def clean_tracks(emb):
    emb = emb.copy()
    emb = emb.loc[:, ~emb.columns.str.startswith("Unnamed")]
    emb["track_id"] = emb["track_id"].map(spotify_track_id)
    emb = emb.dropna(subset=["track_id"]).drop_duplicates("track_id")
    feats = [f for f in C.AUDIO_FEATURES if f in emb.columns]
    for f in feats:
        emb[f] = pd.to_numeric(emb[f], errors="coerce")
    # valid ranges: 0-1 features, tempo > 0
    for f in set(feats) & {"danceability", "energy", "valence", "acousticness",
                           "instrumentalness", "speechiness"}:
        emb.loc[~emb[f].between(0, 1), f] = np.nan
    if "tempo" in emb:
        emb.loc[emb["tempo"] <= 0, "tempo"] = np.nan
    emb = emb.dropna(subset=[f for f in C.CORE_FEATURES if f in emb.columns])
    return emb.reset_index(drop=True)


def link(events, tracks, mapping):
    valid = set(tracks["track_id"])
    cands = events["mbid"].map(lambda m: [s for s in mapping.get(m, []) if s in valid])
    direct = events["direct_spotify_id"].isin(valid)
    n = cands.str.len()
    events["match_status"] = "Unmatched"
    events.loc[n > 1, "match_status"] = "Ambiguous Mapping"
    events.loc[~direct & (n == 1), "match_status"] = "MusicBrainz Mapping"
    events.loc[direct, "match_status"] = "Direct Spotify Match"
    events["track_id"] = pd.NA
    events.loc[direct, "track_id"] = events.loc[direct, "direct_spotify_id"]
    one = events["match_status"].eq("MusicBrainz Mapping")
    events.loc[one, "track_id"] = cands[one].map(lambda v: v[0])

    meta = tracks.rename(columns={"track_name": "embeat_track_name",
                                  "artist_name": "embeat_artist_name"})
    out = events.merge(meta, on="track_id", how="left", validate="many_to_one")
    assert len(out) == len(events), "join changed the number of events"
    linked = out["track_id"].notna()
    out["track_similarity"] = np.nan
    out["artist_similarity"] = np.nan
    out.loc[linked, "track_similarity"] = [similarity(a, b) for a, b in
                                           zip(out.loc[linked, "track_name"], out.loc[linked, "embeat_track_name"])]
    out.loc[linked, "artist_similarity"] = [similarity(a, b) for a, b in
                                            zip(out.loc[linked, "artist_name"], out.loc[linked, "embeat_artist_name"])]
    out["metadata_status"] = "Not Linked"
    out.loc[linked, "metadata_status"] = [metadata_status(t, a) for t, a in
                                          zip(out.loc[linked, "track_similarity"], out.loc[linked, "artist_similarity"])]
    out["model_candidate"] = linked & out["metadata_status"].eq("High Agreement")
    return out


def main():
    listens_raw = load_listens()
    embeat_raw = pd.read_csv(C.EMBEAT_RAW)
    mapping = load_mapping()

    report = ["# Data examination report\n", profile(listens_raw, "ListenBrainz listens (raw)"),
              spelling_variants(listens_raw["artist_name"], "artist_name"),
              profile(embeat_raw, "Embeat tracks (raw)")]

    events, dropped = clean_listens(listens_raw)
    tracks = clean_tracks(embeat_raw)
    integrated = link(events, tracks, mapping)

    status = integrated["match_status"].value_counts()
    meta = integrated["metadata_status"].value_counts()
    report += ["## Cleaning decisions",
               f"- Listens removed (missing key fields or exact duplicates): {dropped}",
               f"- Embeat tracks kept after ID/range checks: {len(tracks):,} of {len(embeat_raw):,}",
               "- Text normalization keeps all Unicode letters (fixes Japanese artist false mismatches)",
               "", "## Join coverage (denominator = all listening events)",
               *[f"- {k}: {v} ({100 * v / len(integrated):.1f}%)" for k, v in status.items()],
               "", "## Metadata agreement",
               *[f"- {k}: {v}" for k, v in meta.items()],
               f"\n**Model candidates: {int(integrated.model_candidate.sum())} / {len(integrated)} "
               f"({100 * integrated.model_candidate.mean():.2f}%)**\n"]

    integrated.to_csv(C.EVENTS_CLEAN, index=False)
    tracks.to_csv(C.TRACKS_CLEAN, index=False)
    C.EXAM_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {C.EVENTS_CLEAN.name} ({len(integrated)} rows) and {C.TRACKS_CLEAN.name} ({len(tracks)} rows)")


if __name__ == "__main__":
    main()
