#!/usr/bin/env bash
# Full AlphaQ pipeline: complete 48-layer Qwen3-Coder-Next.
# Steps: 01 compute alpha → 02 precision solve → 03 GPTQ → 04 eval PPL → 05 inference.
# 01 runs on CPU (minimal config); 02 on CPU; 03–05 require GPU (multi-GPU: device_map=auto is default).
# Log to file: LOG_FILE=logs/run_full.log ./scripts/run_full_pipeline_full.sh

set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee "$LOG_FILE") 2>&1
  echo "=== Logging to $LOG_FILE at $(date) ==="
fi

# Config (edit as needed). If GPTQ OOM: CALIB_BATCH_SIZE=16 or 8; or NSAMPLES=64 SEQLEN=1024
MODEL=${MODEL:-"Qwen/Qwen3-Coder-Next"}
BPP=${BPP:-3.5}
GAMMA=${GAMMA:-10.0}
CANDIDATE_BITS=${CANDIDATE_BITS:-"2,3,4"}
NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-2048}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-32}
DEVICE_MAP=${DEVICE_MAP:-auto}

ALPHA_CSV="${ROOT}/docs/alpha_qwen3_full.csv"
RECIPE_DIR="${ROOT}/docs/bit_recipes"
RECIPE_CSV="${RECIPE_DIR}/qwen3_coder_next_gamma${GAMMA}_bpp${BPP}.csv"
OUT_DIR="${ROOT}/out_qwen3_full"

echo "=== AlphaQ full pipeline (48 layers) ==="
echo "  Model: $MODEL | bpp: $BPP | gamma: $GAMMA | out: $OUT_DIR"
echo ""

# 01: Compute alpha (all 48 layers, CPU)
echo "[1/5] 01_compute_alpha.py (48 layers) ..."
python scripts/01_compute_alpha.py \
  --output_csv "$ALPHA_CSV"

# 02: Precision solve
echo "[2/5] 02_precision_solve.py ..."
python scripts/02_precision_solve.py \
  --alpha_csv "$ALPHA_CSV" \
  --output_dir "$RECIPE_DIR" \
  --bpp "$BPP" \
  --gamma "$GAMMA" \
  --candidate_bits "$CANDIDATE_BITS"

SKIP_GPTQ=${SKIP_GPTQ:-0}
if [[ "$SKIP_GPTQ" == "1" ]]; then
  echo "[3/5] 03 GPTQ skipped (SKIP_GPTQ=1)."
  echo "[4/5] 04_eval_quantized.py on original full model ..."
  python scripts/04_eval_quantized.py \
    --model_path "$MODEL" \
    --seqlen "$SEQLEN" \
    --device_map "$DEVICE_MAP"
  echo "[5/5] 05_inference.py on original full model ..."
  python scripts/05_inference.py \
    --model_path "$MODEL" \
    --prompt "Write a short Python function to add two numbers." \
    --max_new_tokens 128 \
    --device_map "$DEVICE_MAP"
  echo ""
  echo "=== Done (full, skip GPTQ). Alpha: $ALPHA_CSV | Recipe: $RECIPE_CSV ==="
  exit 0
fi

# 03: GPTQ (GPU)
echo "[3/5] 03_gptq_from_recipe.py ..."
python scripts/03_gptq_from_recipe.py \
  --model "$MODEL" \
  --recipe_csv "$RECIPE_CSV" \
  --output_dir "$OUT_DIR" \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
  --calib_batch_size "$CALIB_BATCH_SIZE" \
  --device_map "$DEVICE_MAP"

# 04: Eval PPL
echo "[4/5] 04_eval_quantized.py ..."
python scripts/04_eval_quantized.py \
  --model_path "$OUT_DIR" \
  --seqlen "$SEQLEN" \
  --device_map "$DEVICE_MAP"

# 05: Inference
echo "[5/5] 05_inference.py ..."
python scripts/05_inference.py \
  --model_path "$OUT_DIR" \
  --prompt "Write a short Python function to add two numbers." \
  --max_new_tokens 128 \
  --device_map "$DEVICE_MAP"

echo ""
echo "=== Done (full model). Output model: $OUT_DIR ==="
