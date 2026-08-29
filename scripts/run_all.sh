#!/usr/bin/env bash
# End-to-end ThinkSpark-v2-350M pipeline (Section 14, "train it on any random day").
#
# This orchestrates the stages in order. It is meant to be read and run stage-by-stage
# on Kaggle (9 h/session) — NOT blindly in one go. Comment/uncomment as you progress.
# Every stage is resumable, so a killed session just re-runs the same command.
#
#   conda activate llms
#   cp .env.example .env    # fill in OPENAI_API_KEY / SONIOX_API_KEY / HF_TOKEN
#   bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DG=configs/data_gen.yaml
PC=configs/phase1_corpus.yaml
P1=configs/train_phase1.yaml
P2=configs/train_phase2.yaml

echo "=== 0. sample data (offline, no API) ============================="
python scripts/make_samples.py

echo "=== 1. generate scenarios with the LLM (simple: just run it) ======"
# The plan (Section 8.1-8.3 balance) is auto-built on first run — no separate step
# needed. In a SECOND terminal, watch live progress/cost/latency/pass-fail:
#   python scripts/11_monitor.py --config "$DG"
python scripts/02_generate_scripts.py --config "$DG"

# Advanced / Kaggle 9h-session budget: split the same run across sessions by hand —
#   for part in $(seq 0 7); do
#     python scripts/02_generate_scripts.py --config "$DG" --part "$part"
#   done
#   cat data/scenarios/scenarios_part*.jsonl > data/scenarios/scenarios_all.jsonl

echo "=== 2. chart report (cost, progress, unit-eval — like kupe-tts) ==="
python scripts/12_build_report.py --config "$DG"
python scripts/10_export_costs.py --config "$DG"

echo "=== 3. validate the generated data (Section 8.5) =================="
python scripts/05_validate_data.py --in data/scenarios/scenarios_all.jsonl --judge --judge-n 200

echo "=== 4. render USER audio with Soniox TTS ========================="
python scripts/03_render_user_audio.py --config "$DG" --in data/scenarios/scenarios_all.jsonl

echo "=== 5. Phase-0 encode: audio -> Mimi cb0 + energy/f0 ============="
python scripts/00_encode_audio.py --config "$DG" --audio-dir data/audio --out-dir data/encoded

echo "=== 6. build per-frame training records =========================="
python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl \
    --frames-out data/frames/frames_all.jsonl

echo "=== 6b. refresh chart report + cost export (LLM + TTS, all runs) =="
python scripts/12_build_report.py --config "$DG"
python scripts/10_export_costs.py --config "$DG"

echo "=== 6c. Phase-1 free corpus: fetch + encode + frames (Section 7) ==="
# Common Voice is gated — visit its HF page once and click "Agree" before HF_TOKEN works:
#   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
python scripts/P1_01_fetch_corpus.py --config "$PC" --dry-run   # see the plan first
python scripts/P1_01_fetch_corpus.py --config "$PC"
for lang in en hi gu; do
  python scripts/00_encode_audio.py --config "$DG" --wav-dir "data/phase1_raw/$lang" --out-dir data/encoded
  python scripts/P1_02_build_frames.py --lang "$lang"
done

echo "=== 7. Phase-1 modality alignment (LoRA) ==========================="
python scripts/06_train_phase1.py --config "$P1" --frames "data/frames_phase1/*.jsonl"

echo "=== 8. Phase-2 referee fine-tune ================================="
python scripts/07_train_phase2.py --config "$P2" --frames "data/frames/*.jsonl" \
    --init artifacts/thinkspark-v2-350m/phase1/final/model.pt || \
python scripts/07_train_phase2.py --config "$P2" --frames "data/frames/*.jsonl"

echo "=== 9. evaluate against the Section 10 targets ==================="
python scripts/08_evaluate.py --config "$P2" \
    --ckpt artifacts/thinkspark-v2-350m/phase2/final --frames "data/frames/*.jsonl"

echo "=== done. Wire scripts/09_infer_demo.py into your SDK/LiveKit. ==="
