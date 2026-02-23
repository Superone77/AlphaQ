#!/usr/bin/env bash
# Full AlphaQ pipeline: 2-layer model only (fast, for testing).
# Steps: 01 compute alpha → 02 precision solve → [03 GPTQ → 04 eval PPL → 05 inference].
# 01–02 run on CPU. 03–05 require GPU unless skipped.
# To run on CPU only (no GPU): SKIP_GPTQ=1 ./scripts/run_full_pipeline_2layer.sh
# To log to file: LOG_FILE=logs/run.log ./scripts/run_full_pipeline_2layer.sh
# Or: ./scripts/run_full_pipeline_2layer.sh 2>&1 | tee logs/run.log

set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# Optional: write all output to a log file (and still show in terminal)
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee "$LOG_FILE") 2>&1
  echo "=== Logging to $LOG_FILE at $(date) ==="
fi

# Config (edit as needed)
MODEL=${MODEL:-"Qwen/Qwen3-Coder-Next"}
BPP=${BPP:-3.5}
GAMMA=${GAMMA:-10.0}
CANDIDATE_BITS=${CANDIDATE_BITS:-"2,3,4"}
NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-2048}
DEVICE_MAP=${DEVICE_MAP:-auto}
# Set SKIP_GPTQ=1 to run only 01 and 02 (CPU-only test)
SKIP_GPTQ=${SKIP_GPTQ:-0}

ALPHA_CSV="${ROOT}/docs/alpha_qwen3_2layer.csv"
RECIPE_DIR="${ROOT}/docs/bit_recipes_2layer"
RECIPE_CSV="${RECIPE_DIR}/qwen3_coder_next_gamma${GAMMA}_bpp${BPP}.csv"
OUT_DIR="${ROOT}/out_qwen3_2layer"

echo "=== AlphaQ full pipeline (2 layers) ==="
echo "  Model: $MODEL | bpp: $BPP | gamma: $GAMMA | out: $OUT_DIR"
echo ""

# 01: Compute alpha (2 layers only, CPU)
echo "[1/5] 01_compute_alpha.py (max_layers=2) ..."
python scripts/01_compute_alpha.py \
  --output_csv "$ALPHA_CSV" \
  --max_layers 2

# 02: Precision solve
echo "[2/5] 02_precision_solve.py ..."
python scripts/02_precision_solve.py \
  --alpha_csv "$ALPHA_CSV" \
  --output_dir "$RECIPE_DIR" \
  --bpp "$BPP" \
  --gamma "$GAMMA" \
  --candidate_bits "$CANDIDATE_BITS"
if [[ "$SKIP_GPTQ" == "1" ]]; then
  echo "[3/5] 03 GPTQ skipped (SKIP_GPTQ=1)."
  echo "[4/5] 04_eval_quantized.py on original 2-layer model ..."
  python scripts/04_eval_quantized.py \
    --from_config --max_layers 2 \
    --seqlen "$SEQLEN" \
    --device_map "$DEVICE_MAP"
  echo "[5/5] 05_inference.py on original 2-layer model ..."
  python scripts/05_inference.py \
    --from_config --max_layers 2 \
    --prompt "Write a short Python function to add two numbers." \
    --max_new_tokens 128 \
    --device_map "$DEVICE_MAP"
  echo ""
  echo "=== Done (2-layer, skip GPTQ). Alpha: $ALPHA_CSV | Recipe: $RECIPE_CSV ==="
  exit 0
fi

# 03: GPTQ (GPU; recipe has 2 layers → only first 2 layers' experts quantized)
echo "[3/5] 03_gptq_from_recipe.py ..."
python scripts/03_gptq_from_recipe.py \
  --model "$MODEL" \
  --recipe_csv "$RECIPE_CSV" \
  --output_dir "$OUT_DIR" \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
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
echo "=== Done (2-layer). Output model: $OUT_DIR ==="
