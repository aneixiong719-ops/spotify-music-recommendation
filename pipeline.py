"""End-to-end pipeline used by GitHub Actions.

ingest new batch -> clean & join -> ML indicators -> user indicators -> report

Run:  python pipeline.py
"""
import json
from datetime import datetime, timezone

import pandas as pd

import config as C
import data_exam
import indicators
import ml_models


def ingest_batch():
    """Append the rows of batch.csv (written by batch_sim.py) to the stream file."""
    if not C.BATCH_FILE.exists():
        return 0
    batch = pd.read_csv(C.BATCH_FILE)
    batch = batch.loc[:, ~batch.columns.str.startswith("Unnamed")]
    if batch.empty:
        return 0
    if C.LISTENS_STREAM.exists():
        stream = pd.concat([pd.read_csv(C.LISTENS_STREAM), batch], ignore_index=True)
    else:
        stream = batch
    stream = stream.drop_duplicates(["user_id", "listened_at", "track_name", "artist_name"])
    stream.to_csv(C.LISTENS_STREAM, index=False)
    return len(batch)


def write_report(n_new):
    users = pd.read_csv(C.USER_INDICATORS)
    events = pd.read_csv(C.EVENTS_CLEAN)
    m = json.loads(C.MODEL_METRICS.read_text())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Automated pipeline report", f"_Generated {now} — new rows in this batch: {n_new}_", "",
             "## Data", f"- Listening events: {len(events):,} from {events.user_id.nunique()} users",
             f"- Linked to Embeat: {events.track_id.notna().mean():.1%}",
             f"- Model candidates (high metadata agreement): {events.model_candidate.mean():.1%}", "",
             "## Q1 — Listener taste profiles", users.to_markdown(index=False), "",
             "## Q2 — Recommendation quality (chronological hold-out)"]
    if "models" in m:
        lines += [f"Users evaluated: {m['users_evaluated']} of {m['users_total']} (K = {m['k']})", "",
                  pd.DataFrame(m["models"]).T.to_markdown()]
    else:
        lines.append(m.get("note", ""))
    (C.REPORTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    n_new = ingest_batch()
    data_exam.main()
    ml_models.main()
    indicators.main()
    write_report(n_new)
    print(f"Pipeline finished. New rows ingested: {n_new}")


if __name__ == "__main__":
    main()
