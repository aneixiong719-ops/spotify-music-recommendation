"""Batch simulator (adapted from the course's BatchLoader).

Simulates new ListenBrainz listens arriving over time: rows are released in
chronological order, written to batch.csv and pushed to GitHub. Each push
triggers the GitHub Actions pipeline (.github/workflows/pipeline.yml).

Usage:  python batch_sim.py data/raw/listenbrainz_new_listens.csv --batch-size 50 --n-batches 3
"""
import argparse
import time

import pandas as pd
from git import Repo

PATH_OF_GIT_REPO = r"./.git"
COMMIT_MESSAGE = "new batch"


class BatchLoader:
    def __init__(self, data_file, batch_size) -> None:
        self.data_file = data_file
        self.data = None
        self.batch_size = batch_size
        self.idx = 0

    def generate(self):
        if self.data is None:
            # release listens in the order they happened, like a real stream
            self.data = pd.read_csv(self.data_file).sort_values("listened_at").reset_index(drop=True)
        end = min(self.idx + self.batch_size, self.data.shape[0])
        selected = self.data.iloc[self.idx:end]
        self.idx = end
        selected.to_csv("batch.csv", index=False)
        return len(selected)


def git_push():
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add("batch.csv")
        repo.index.commit(COMMIT_MESSAGE)
        repo.remote(name="origin").pull(rebase=True)   # Actions may have pushed results meanwhile
        repo.remote(name="origin").push()
    except Exception as exc:
        print("Some error occurred while pushing the batch:", exc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--wait", type=int, default=120, help="seconds between batches")
    args = ap.parse_args()

    bl = BatchLoader(args.data_file, args.batch_size)
    for _ in range(args.n_batches):
        if bl.generate() == 0:
            break
        git_push()
        time.sleep(args.wait)
