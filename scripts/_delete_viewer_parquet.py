"""One-off: delete the old Dataset Viewer parquet (data/train-00000-of-00002.parquet,
data/train-00001-of-00002.parquet) from the Phase-2 HF repo — superseded by
scenarios/scenarios_all.jsonl + data/phase2-shard-*.parquet. Run once, then delete this file.

    python scripts/_delete_viewer_parquet.py
"""
import os
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from huggingface_hub import HfApi, CommitOperationDelete

repo = "anuj-inavlabs/Thinkspark-v2-270m-training-data"
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

targets = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
print(f"deleting {targets} from {repo} ...")
api.create_commit(
    repo_id=repo,
    repo_type="dataset",
    operations=[CommitOperationDelete(path_in_repo=t) for t in targets],
    commit_message="remove old Dataset Viewer parquet (superseded by scenarios_all.jsonl + phase2-shard parquet)",
)
print("done.")
