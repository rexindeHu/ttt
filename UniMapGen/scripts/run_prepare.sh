#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-$ROOT/dataset}
INPUT_JSONL=${INPUT_JSONL:-$DATASET_ROOT/train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/prepare}
OUTPUT_JSONL=${OUTPUT_JSONL:-$OUTPUT_DIR/train_tokens.jsonl}
SUMMARY_JSON=${SUMMARY_JSON:-$OUTPUT_DIR/prepare_summary.json}

IMAGE_SIZE=${IMAGE_SIZE:-896}
COORD_NUM_BINS=${COORD_NUM_BINS:-896}
TOKEN_SCHEMA=${TOKEN_SCHEMA:-shared_numbers}
CATEGORIES=${CATEGORIES:-auto}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
DISABLE_LEGACY_TEXT_PROMPT_TOKENS=${DISABLE_LEGACY_TEXT_PROMPT_TOKENS:-1}
STRICT=${STRICT:-0}

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "[prepare] missing input jsonl: $INPUT_JSONL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  "$PYTHON_BIN"
  "$ROOT/scripts/prepare_tokens.py"
  --input-jsonl "$INPUT_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --summary-json "$SUMMARY_JSON"
  --image-size "$IMAGE_SIZE"
  --coord-num-bins "$COORD_NUM_BINS"
  --token-schema "$TOKEN_SCHEMA"
  --categories "$CATEGORIES"
  --max-seq-len "$MAX_SEQ_LEN"
)

if [[ "$DISABLE_LEGACY_TEXT_PROMPT_TOKENS" == "1" ]]; then
  cmd+=(--disable-legacy-text-prompt-tokens)
fi
if [[ "$STRICT" == "1" ]]; then
  cmd+=(--strict)
fi

exec "${cmd[@]}"
