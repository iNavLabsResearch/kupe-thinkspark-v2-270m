from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete

api = HfApi()
repo = "anuj-inavlabs/kupe-thinkspark-audio-270m"
dest = "phase2/runs/20260902-103400"

moves = {
    "best": f"{dest}/best",
    "final": f"{dest}/final",
    "step1000": f"{dest}/step1000",
    "step1500": f"{dest}/step1500",
    "step2000": f"{dest}/step2000",
    "reports/eval_phase2.json": f"{dest}/reports/eval_phase2.json",
    "reports/eval_phase1.json": "phase1/reports/eval_phase1.json",
}

files = api.list_repo_files(repo, repo_type="model")
ops = []
for src_prefix, dst_prefix in moves.items():
    matched = [f for f in files if f == src_prefix or f.startswith(src_prefix + "/")]
    for f in matched:
        new_path = dst_prefix + f[len(src_prefix):]
        ops.append((f, new_path))

commit_ops = []
for src, dst in ops:
    commit_ops.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst))
    commit_ops.append(CommitOperationDelete(path_in_repo=src))

if not commit_ops:
    print("Nothing matched — check file list / prefixes.")
else:
    api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=commit_ops,
        commit_message="reorganize: nest checkpoints/reports under phase2/runs/20260902-103400",
    )
    print(f"Moved {len(ops)} paths.")
