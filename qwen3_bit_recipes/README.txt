Place AlphaQ recipe CSV files here (name, bit_width), or use the output of:
  python scripts/02_precision_solve.py --alpha_csv docs/alpha_qwen3_full.csv --output_dir docs/bit_recipes --bpp 4.0 --gamma 10.0

Example filename: qwen3_coder_next_gamma10.0_bpp4.0.csv
Then run GPTQ with:
  python scripts/03_gptq_from_recipe.py --recipe_csv qwen3_bit_recipes/qwen3_coder_next_gamma10.0_bpp4.0.csv --output_dir ./out_qwen3_bpp4 ...
