#!/usr/bin/env bash
# Render the WHOLE consolidated Phase-2 corpus in one run — so you can start this once
# and walk away. short + extended + long were originally three separate corpora with
# their own scenario/audio dirs; they've since been merged (verified: 0 scenario_id /
# wav-filename collisions between any of them) into ONE file/dir:
#   data/scenarios/scenarios_all.jsonl  +  data/audio/
#
# 03_render_user_audio.py is independently resumable (skips clips already on disk) — a
# scenario whose wav is already rendered (from ANY of the original corpora) is skipped;
# whatever isn't done yet renders. Killing and restarting this just picks up where it
# left off — nothing is re-rendered or duplicated. A real audio-hours summary, broken
# down by length band, prints at the end (scripts/17_audio_summary.py).
#
#   conda activate llms
#   bash scripts/18_render_all.sh                # live dashboard, stay attached
#   nohup bash scripts/18_render_all.sh --quiet > render_all.log 2>&1 &   # unattended:
#                                                 #   run this, then go to sleep; check
#                                                 #   progress any time with:
#                                                 #     tail -f render_all.log
set -uo pipefail
cd "$(dirname "$0")/.."

# A plain string, not an array: macOS ships bash 3.2, which throws "unbound variable"
# on an EMPTY array expansion under `set -u` — a single flag word doesn't have that
# problem and needs no quoting gymnastics either.
QUIET_FLAG=""
if [[ "${1:-}" == "--quiet" ]]; then
  QUIET_FLAG="--quiet"
fi

CFG="configs/data_gen.yaml"
IN="data/scenarios/scenarios_all.jsonl"
AUDIO_DIR="data/audio"

echo "================================================================"
echo "  rendering: full consolidated corpus   ($CFG)"
echo "================================================================"

if [[ ! -f "$IN" ]]; then
  echo "ERROR: $IN not found." >&2
  exit 1
fi

python scripts/03_render_user_audio.py --config "$CFG" --in "$IN" \
    --audio-dir "$AUDIO_DIR" $QUIET_FLAG
status=$?

echo ""
echo "================================================================"
echo "  render_all.sh summary"
echo "================================================================"
if [[ $status -eq 0 ]]; then
  echo "  done (exit 0)"
else
  echo "  FAILED (exit $status) — see output above; re-run this script to retry, it will resume"
fi
echo ""
python scripts/17_audio_summary.py
exit $status
