"""Machine-learning indicators.

(a) Mood clusters: KMeans on standardized audio features gives every track a
    'mood' label (e.g. calm/acoustic vs. energetic/danceable).
(b) Content-based recommender: a user's taste vector = mean audio profile of
    the tracks they played; candidates are ranked by cosine similarity.
(c) Fair evaluation: chronological hold-out (each user's most recent 20% of
    distinct tracks), compared with a popularity baseline using
    Precision@K, Recall@K, NDCG@K and catalog coverage.

Run:  python ml_models.py   (after data_exam.py)
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config as C


def feature_cols(tracks):
    return [f for f in C.AUDIO_FEATURES if f in tracks.columns]


def mood_clusters(tracks):
    feats = feature_cols(tracks)
    X = StandardScaler().fit_transform(tracks[feats].fillna(tracks[feats].median()))
    k = min(C.N_MOOD_CLUSTERS, len(tracks))
    km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X)
    tracks = tracks.copy()
    tracks["mood_cluster"] = km.labels_
    # human-readable name from the cluster's strongest traits
    centers = tracks.groupby("mood_cluster")[["energy", "valence", "danceability"]].mean()
    def name(row):
        e = "High-energy" if row.energy >= tracks.energy.median() else "Calm"
        v = "upbeat" if row.valence >= tracks.valence.median() else "moody"
        d = "dance" if row.danceability >= tracks.danceability.median() else "listening"
        return f"{e} {v} {d}"
    names = centers.apply(name, axis=1)
    tracks["mood_name"] = tracks["mood_cluster"].map(names)
    return tracks, X


def chronological_split(events, test_share=0.2, min_tracks=5):
    """Per user: first-listen order of distinct tracks; last 20% become test."""
    first = (events.sort_values("listened_at")
             .drop_duplicates(["user_id", "track_id"]))
    train, test = [], []
    for user, g in first.groupby("user_id"):
        if len(g) < min_tracks:
            continue                       # not enough history to evaluate fairly
        n_test = max(1, int(round(len(g) * test_share)))
        train.append(g.iloc[:-n_test])
        test.append(g.iloc[-n_test:])
    if not train:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train), pd.concat(test)


def scale_matrix(tracks):
    feats = feature_cols(tracks)
    X = StandardScaler().fit_transform(tracks[feats].fillna(tracks[feats].median()))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return pd.DataFrame(X, index=tracks["track_id"])


def content_recs(user_tracks, Xn, k):
    seen = set(user_tracks)
    profile = Xn.loc[list(seen)].mean().values
    profile = profile / (np.linalg.norm(profile) + 1e-9)
    scores = pd.Series(Xn.values @ profile, index=Xn.index).drop(list(seen), errors="ignore")
    return scores.nlargest(k)


def popularity_recs(train, user_tracks, k):
    counts = train.groupby("track_id")["user_id"].count().sort_values(ascending=False)
    return [t for t in counts.index if t not in set(user_tracks)][:k]


def metrics_at_k(rec, relevant, k):
    rec, relevant = list(rec)[:k], set(relevant)
    hits = [1 if r in relevant else 0 for r in rec]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return {"precision": sum(hits) / k, "recall": sum(hits) / max(len(relevant), 1),
            "ndcg": dcg / idcg if idcg else 0.0}


def evaluate(events, tracks):
    Xn = scale_matrix(tracks)
    train, test = chronological_split(events)
    if train.empty:
        return {"note": "No user has enough history for a fair hold-out yet."}, []
    rows, rec_sets = [], {"content": set(), "popularity": set()}
    for user, t in test.groupby("user_id"):
        hist = train.loc[train.user_id == user, "track_id"]
        relevant = t["track_id"]
        c = content_recs(hist, Xn, C.K).index
        p = popularity_recs(train, hist, C.K)
        rec_sets["content"] |= set(c)
        rec_sets["popularity"] |= set(p)
        for model, rec in (("content_based", c), ("popularity", p)):
            rows.append({"user_id": user, "model": model, **metrics_at_k(rec, relevant, C.K)})
    per_user = pd.DataFrame(rows)
    summary = per_user.groupby("model")[["precision", "recall", "ndcg"]].mean().round(4).to_dict("index")
    summary["content_based"]["catalog_coverage"] = round(len(rec_sets["content"]) / len(tracks), 4)
    summary["popularity"]["catalog_coverage"] = round(len(rec_sets["popularity"]) / len(tracks), 4)
    result = {"k": C.K, "users_evaluated": int(per_user.user_id.nunique()),
              "users_total": int(events.user_id.nunique()),
              "split": "chronological, last 20% of each user's distinct tracks",
              "models": summary}
    return result, per_user.to_dict("records")


def main():
    events = pd.read_csv(C.EVENTS_CLEAN, parse_dates=["listened_at"])
    tracks = pd.read_csv(C.TRACKS_CLEAN)
    tracks, _ = mood_clusters(tracks)
    tracks.to_csv(C.TRACKS_CLEAN, index=False)         # add mood indicator to the dataset

    model_events = events[events.model_candidate].copy()
    result, per_user = evaluate(model_events, tracks)
    result["per_user"] = per_user
    C.MODEL_METRICS.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # final recommendations for every user, trained on full history
    Xn = scale_matrix(tracks)
    names = tracks.set_index("track_id")
    recs = []
    for user, g in model_events.groupby("user_id"):
        for rank, (tid, score) in enumerate(content_recs(g["track_id"], Xn, C.K).items(), 1):
            recs.append({"user_id": user, "rank": rank, "track_id": tid,
                         "track_name": names.at[tid, "track_name"] if "track_name" in names else tid,
                         "artist_name": names.at[tid, "artist_name"] if "artist_name" in names else "",
                         "mood_name": names.at[tid, "mood_name"], "similarity": round(score, 4)})
    pd.DataFrame(recs).to_csv(C.RECOMMENDATIONS, index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "per_user"}, indent=2))


if __name__ == "__main__":
    main()
