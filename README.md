# Paper Code

This directory preserves the broader AlphaQ implementation used for the manuscript:

- `main.py`: Mixtral-style MoE quantization entry point.
- `deepseek_main.py`: DeepSeek MoE entry point.
- `qwen_main.py`: Qwen MoE entry point.
- `utils_alpha.py`: PL Alpha Hill and FARMS-based alpha estimation.
- `precision_solver.py`: budget-constrained mixed-precision solver.
- `evaluate_quantized*.py` and `inference*.py`: evaluation and generation utilities.
- `alpha_*` and `var_quantization_noise/`: figure and analysis utilities used during the paper revision.

The repository root contains the newer Qwen3-Coder-Next pipeline scripts under
`scripts/`. Use this directory when reproducing the paper-era multi-model AlphaQ
experiments or comparing against the manuscript code path.

Install the paper-code dependencies from this directory:

```bash
cd paper_code
pip install -r requirements.txt
```

Example AlphaQ allocation run:

```bash
python main.py <MODEL_PATH> \
  --wbits 2bit \
  --attn_bits 4bit \
  --dataset wikitext2 \
  --groupsize 128 \
  --eval_ppl \
  --mixed_type no_calib_auto_programming \
  --cache_dir ./alpha_cache \
  --milp_candidate_bits 1,2,3,4 \
  --milp_bpp_budget 2.5 \
  --milp_gamma 1.0
```
