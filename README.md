# AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization

Paper: https://arxiv.org/abs/2606.04980
Project Page: https://superone77.github.io/AlphaQ/

AlphaQ is a novel **calibration-free** bit-allocation method for Mixture-of-Experts (MoE) model quantization. Unlike traditional data-driven methods that rely on calibration data to estimate expert importance, AlphaQ derives importance signals directly from model weights by leveraging **Heavy-Tailed Self-Regularization (HT-SR) theory**.

## Key Features

- **Calibration-Free**: No calibration data required for bit allocation, avoiding domain bias
- **Data-Independent**: Derives expert importance from weight matrix spectral properties
- **Global Optimization**: Formulates mixed-precision quantization as a budget-constrained Integer Linear Programming (ILP) problem
- **Fine-Grained Allocation**: Supports layer-wise bit allocation within experts for optimal precision distribution
- **FARMS Support**: Fixed Aspect Ratio Matrix Sampling for robust alpha estimation across varying matrix shapes

## Methodology

AlphaQ is based on the principle that experts with heavier-tailed weight spectra (smaller α values in PL Alpha Hill metric) tend to be better trained and more sensitive to quantization. The method:

1. **Computes PL Alpha Hill** for each layer's weight matrix using either:
   - **Baseline Mode**: Full SVD on entire weight matrices
   - **FARMS Mode**: Fixed Aspect Ratio Matrix Sampling for aspect-ratio-robust estimation

2. **Formulates ILP Optimization**:
   - Minimizes total importance-weighted quantization error
   - Subject to global bit-budget constraint
   - Assigns higher precision to layers with heavier-tailed spectra (lower α)

3. **Solves Optimal Bit Assignment** using PuLP solver

## Supported Models

- **Mixtral-8×7B**
- **DeepSeekV2-Lite**
- **Qwen1.5-MoE**

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.2.1
- Transformers >= 4.36.1
- PuLP (for ILP solver)
- NumPy, Pandas, SciPy
- Accelerate, Einops

## Usage

### AlphaQ with MILP-based Automatic Bit Allocation

This is the recommended mode that implements the full AlphaQ pipeline:

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
    --milp_gamma 1.0 \
    --pack \
    --save \
    --saving_path <OUTPUT_PATH>
```

#### Key Arguments for AlphaQ Mode

| Argument | Description |
|----------|-------------|
| `--mixed_type no_calib_auto_programming` | Enable AlphaQ calibration-free bit allocation |
| `--milp_candidate_bits` | Candidate bit-widths for MILP solver (e.g., `2,3,4`) |
| `--milp_bpp_budget` | Average bit-width budget (e.g., `2.5`, `3.5`) |
| `--milp_gamma` | Gamma parameter for sensitivity weighting (default: `1.0`) |
| `--cache_dir` | Directory to cache computed alpha values |

### Alpha-based-Importance-only Mixed Precision (Ratio Mode)

An alternative mode using alpha-based expert ranking:

```bash
python main.py <MODEL_PATH> \
    --wbits 2bit \
    --attn_bits 4bit \
    --dataset wikitext2 \
    --groupsize 128 \
    --eval_ppl \
    --mixed_type mixed_with_alpha \
    --cache_dir ./alpha_cache \
    --high_bit_experts_ratio 0.25 \
    --low_bit_experts_ratio 0.25
```

### Using FARMS Mode

Enable FARMS for robust alpha estimation on rectangular weight matrices:

```bash
export ALPHA_MODE=FARMS
export FARMS_M_SUB=128
export FARMS_N_SUB=128
export FARMS_MAX_BLOCKS=256

python main.py <MODEL_PATH> \
    --mixed_type no_calib_auto_programming \
    --milp_bpp_budget 2.5 \
    ...
```

#### FARMS Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPHA_MODE` | `FARMS` | `FARMS` or `BASELINE` |
| `FARMS_M_SUB` | `128` | Sub-matrix row dimension |
| `FARMS_N_SUB` | `128` | Sub-matrix column dimension |
| `FARMS_STRIDE_M` | Same as M_SUB | Row stride for sampling |
| `FARMS_STRIDE_N` | Same as N_SUB | Column stride for sampling |
| `FARMS_MAX_BLOCKS` | `256` | Maximum sub-matrices to sample |

## Quantization Modes

| Mode | Description |
|------|-------------|
| `no_calib_auto_programming` | **AlphaQ**: MILP-based calibration-free bit allocation |
| `mixed_with_alpha` | Alpha-based expert ranking with fixed ratios |
| `uniform` | All experts at same precision |
| `manual` | Pre-defined precision from file |
| `random` | Random expert bit-width selection |



## Code Structure

```
AlphaQ/
├── main.py                 # Main entry point for Mixtral quantization
├── deepseek_main.py        # Entry point for DeepSeek models
├── qwen_main.py            # Entry point for Qwen models
├── utils_alpha.py          # Alpha (PL Alpha Hill) computation with FARMS
├── precision_solver.py     # ILP solver for bit allocation
├── gptq.py                 # GPTQ quantization implementation
├── modelutils.py           # Model utilities
├── datautils.py            # Data loading utilities
├── eval_utils.py           # Evaluation utilities
├── eval_ppl_utils.py       # Perplexity evaluation
├── models/                 # Model layer implementations
├── quant/                  # Quantization kernels and utilities
├── utils/                  # Quantizer implementations
└── deepseek_moe/           # DeepSeek MoE model configuration
```




## Acknowledgments

AlphaQ builds upon Heavy-Tailed Self-Regularization (HT-SR) theory and the FARMS methodology for robust spectral analysis. We thank the authors of these foundational works for their contributions.
