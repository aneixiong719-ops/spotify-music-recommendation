"""Indicators that answer the two business questions.

Q1  What does each listener's taste look like, and how varied is it?
    -> taste profile (mean audio features), taste consistency (feature spread),
       repeat share, artist diversity, dominant mood cluster.
Q2  Can audio features recommend songs a listener will actually play next?
    -> data readiness per user (linked share, eligible tracks) + model results
       from ml_models.py (Precision/Recall/NDCG@10 vs popularity baseline).

The indicators are added back to the event table (enriched file) and also
saved as one row per user.  Run:  python indicators.py
"""
import pandas as pd

import config as C


def main():
    events = pd.read_csv(C.EVENTS_CLEAN, parse_dates=["listened_at"])
    tracks = pd.read_csv(C.TRACKS_CLEAN)
    feats = [f for f in C.CORE_FEATURES if f in tracks.columns]

    # ---- event-level indicators (integrated into the dataframe) ----
    events = events.drop(columns=[c for c in ("mood_cluster", "mood_name") if c in events], errors="ignore")
    events = events.merge(tracks[["track_id", "mood_cluster", "mood_name"]], on="track_id", how="left")
    events["is_repeat"] = events.duplicated(["user_id", "track_name", "artist_name"])
    events["hour"] = events["listened_at"].dt.hour
    events["weekday"] = events["listened_at"].dt.day_name()
    events.to_csv(C.EVENTS_CLEAN, index=False)

    # ---- user-level indicators ----
    g = events.groupby("user_id")
    users = pd.DataFrame({
        "listens": g.size(),
        "unique_tracks": g.apply(lambda d: d[["track_name", "artist_name"]].drop_duplicates().shape[0]),
        "unique_artists": g["artist_name"].nunique(),
        "repeat_share": g["is_repeat"].mean().round(3),
        "linked_share": g["track_id"].apply(lambda s: s.notna().mean()).round(3),
        "eligible_events": g["model_candidate"].sum(),
    })
    users["artist_diversity"] = (users.unique_artists / users.listens).round(3)

    elig = events[events.model_candidate]
    if not elig.empty:
        profile = elig.groupby("user_id")[feats].mean().round(3).add_prefix("taste_")
        unit = [f for f in feats if f != "tempo"]          # 0-1 features only, same scale
        spread = elig.groupby("user_id")[unit].std().mean(axis=1).round(3).rename("taste_spread")
        mood = elig.groupby("user_id")["mood_name"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
        users = users.join(profile).join(spread).join(mood.rename("dominant_mood"))
    users["eligible_unique_tracks"] = elig.groupby("user_id")["track_id"].nunique()
    users = users.fillna({"eligible_unique_tracks": 0}).reset_index()
    users.to_csv(C.USER_INDICATORS, index=False)
    print(users.to_string(index=False))


if __name__ == "__main__":
    main()
