#!/usr/bin/env python
"""
Evaluation script for DeepSeek MoE quantized models using lm_eval.

Loads a quantized DeepSeek model exported by deepseek_main.py using
the local deepseek_moe model definitions, then evaluates with lm_eval.
"""

import argparse
import os
import torch
import torch.nn as nn

from transformers import AutoTokenizer
from accelerate import init_empty_weights

# Weight IO reused from pack/inference utilities
from inference import load_weights

# Use local DeepSeek model definitions
from deepseek_moe.configuration_deepseek import DeepseekV2Config
from deepseek_moe.modeling_deepseek import DeepseekV2ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized DeepSeek MoE models using lm_eval"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model directory (contains qmodel.pt and config.json)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu,hellaswag,winogrande,arc,truthfulqa",
        help="Comma-separated list of tasks to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (e.g., cuda or cpu)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use for the model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--evaluate_bf16",
        action="store_true",
        help="If set, load the original (non-quantized) model in bfloat16 like deepseek_main.py and evaluate.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Placement strategy for quantized path. Use 'auto' to shard across multiple GPUs with accelerate.",
    )
    return parser.parse_args()


def _torch_dtype_from_str(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    return torch.float16


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module):
    """
    Replace submodule by dotted path on root with new_module.
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def load_quantized_deepseek(
    save_dir: str,
    attn_implementation: str = "eager",
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
    device_map: str = "none",
):
    """
    Load a DeepSeek quantized model saved by deepseek_main.py using local deepseek_moe definitions.
    This loader maps saved module names directly to model modules, without relying on Mixtral-specific patchers.
    """
    # 1) Build model skeleton with DeepSeek classes
    config = DeepseekV2Config.from_pretrained(save_dir, trust_remote_code=True)
    with init_empty_weights():
        # Some custom implementations expose _from_config instead of from_config
        model = DeepseekV2ForCausalLM._from_config(  # type: ignore[attr-defined]
            config,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )

    # 2) Load serialized quantized weights
    weights = load_weights(save_dir, map_location="cpu")

    use_auto_shard = (device_map == "auto") and torch.cuda.is_available() and torch.cuda.device_count() > 1
    # When auto sharding, first materialize on CPU and later dispatch to GPUs
    target_device = "cpu" if use_auto_shard else (device if device.startswith("cuda") or device == "cpu" else "cuda")

    # 3) Name modules and apply weights
    # Ensure each module has .name for convenience
    for name, module in model.named_modules():
        module.name = name  # type: ignore[attr-defined]

    @torch.no_grad()
    def _apply_state_to_module(module: nn.Module, state_dict: dict):
        # If it is a quantized linear represented by QLinear state
        if "W_q" in state_dict:
            from quant.QLinear import QLinear
            qlinear = QLinear(
                quant_config=None,
                compute_dtype=compute_dtype,
                device=target_device,
            )
            qlinear.load_state_dict(state_dict)
            return qlinear
        # Otherwise assign parameters
        for key, tensor in state_dict.items():
            setattr(
                module,
                key,
                nn.Parameter(
                    tensor.to(device=target_device, dtype=compute_dtype, non_blocking=True),
                    requires_grad=False,
                ),
            )
        return module

    # 4) Replace/assign modules by saved names
    # Iterate over a static list of names to avoid runtime mutation issues
    module_lookup = dict(model.named_modules())
    for name, state in weights.items():
        if name not in module_lookup:
            continue
        current = module_lookup[name]
        new_module = _apply_state_to_module(current, state)
        if new_module is not current:
            _set_module(model, name, new_module)

    # 5) Placement
    if use_auto_shard:
        try:
            from accelerate.utils import infer_auto_device_map, get_balanced_memory
            from accelerate import dispatch_model
            print("Using accelerate to shard model across GPUs (device_map=auto)")
            max_memory = get_balanced_memory(model, dtype=compute_dtype)
            inferred_map = infer_auto_device_map(model, max_memory=max_memory, dtype=compute_dtype, no_split_module_classes=None)
            model = dispatch_model(model, device_map=inferred_map)
        except Exception as e:
            print(f"Auto sharding failed ({e}). Falling back to single device placement.")
            model.to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=compute_dtype, non_blocking=True)
    else:
        model.to(device=target_device, dtype=compute_dtype, non_blocking=True)
    model.eval()
    return model


def main():
    args = parse_args()

    print("=" * 80)
    print("Quantized DeepSeek Model Evaluation")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Attention impl: {args.attn_implementation}")
    print(f"Dtype: {args.dtype}")
    print(f"Evaluate bf16 original model: {args.evaluate_bf16}")
    print(f"Device map (quantized path): {args.device_map}")
    print("=" * 80)

    # Two evaluation modes:
    # - Quantized evaluation (default): load from quantized directory using serialized qmodel
    # - BF16 evaluation (--evaluate_bf16): load original model like deepseek_main.py and evaluate in bf16
    if args.evaluate_bf16:
        print(f"\nLoading bf16 original DeepSeek model from: {args.model_path}")
        config = DeepseekV2Config.from_pretrained(
            args.model_path,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        # Mirror deepseek_main.py behavior but use bfloat16 as requested
        model = DeepseekV2ForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            device_map="auto" if args.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = _torch_dtype_from_str(args.dtype)
        print(f"\nLoading quantized DeepSeek model from: {args.model_path}")
        model = load_quantized_deepseek(
            args.model_path,
            attn_implementation=args.attn_implementation,
            device=args.device,
            compute_dtype=compute_dtype,
            device_map=args.device_map,
        )
        model.eval()
        # Load tokenizer saved alongside quantized model
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare tasks
    task_list = [task.strip() for task in args.tasks.split(",") if task.strip()]

    # lm_eval integration
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    print("Wrapping model with HFLM...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Evaluating on tasks: {task_list}")
    for task in task_list:
        print(f"\nEvaluating task: {task}")
        print("-" * 80)
        result = evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            batch_size=args.batch_size,
        )
        if "results" in result and task in result["results"]:
            task_results = result["results"][task]
            print(f"\n{task} Results:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Evaluation script for DeepSeek MoE quantized models using lm_eval.

This script loads a quantized DeepSeek model exported by deepseek_main.py
and evaluates it on various tasks using lm_eval.
"""

import argparse
import os
import torch

from transformers import AutoTokenizer

# Reuse quantized model loading utilities
from inference import load_weights, patch_model, setup_model
from accelerate import init_empty_weights

# DeepSeek model definitions (local)
from deepseek_moe.configuration_deepseek import DeepseekV2Config
from deepseek_moe.modeling_deepseek import DeepseekV2ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized DeepSeek MoE models using lm_eval"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model directory (contains qmodel.pt and config.json)",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu,hellaswag,winogrande,arc,truthfulqa",
        help="Comma-separated list of tasks to evaluate on",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (e.g., cuda or cpu)",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use for the model",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype",
    )

    return parser.parse_args()


def _torch_dtype_from_str(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    return torch.float16


def load_quantized_deepseek(save_dir: str, attn_implementation: str = "eager", device: str = "cuda", compute_dtype: torch.dtype = torch.float16):
    """
    Load a DeepSeek quantized model saved by deepseek_main.py using local deepseek_moe definitions.
    """
    # 1) Build model skeleton with DeepSeek classes
    config = DeepseekV2Config.from_pretrained(save_dir, trust_remote_code=True)
    
    with init_empty_weights():
        model = DeepseekV2ForCausalLM.from_config(
            config,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )

    # 2) Prepare for patching
    model.save_dir = save_dir
    setup_model(model)

    # 3) Load serialized quantized weights
    weights = load_weights(save_dir, map_location="cpu")

    # 4) Define how to materialize each module on the target device/dtype
    @torch.no_grad()
    def _load_module(module, params=None):
        nonlocal device, compute_dtype
        target_device = device if device.startswith("cuda") or device == "cpu" else "cuda"
        if module.name not in weights:
            return module.to(device=target_device, dtype=compute_dtype, non_blocking=True)
        state_dict = weights[module.name]
        # If it is a quantized linear layer stored as QLinear state dict
        if "W_q" in state_dict:
            from quant.QLinear import QLinear
            module = QLinear(
                quant_config=None,
                compute_dtype=compute_dtype,
                device=target_device,
            )
            module.load_state_dict(state_dict)
        else:
            for key in state_dict:
                setattr(
                    module,
                    key,
                    torch.nn.Parameter(
                        state_dict[key].to(device=target_device, dtype=compute_dtype, non_blocking=True),
                        requires_grad=False,
                    ),
                )
        return module

    # 5) Patch the whole model tree
    patch_model(model, _load_module, _load_module, {k: None for k in getattr(model, "linear_tags", [])})

    return model


def main():
    args = parse_args()

    print("=" * 80)
    print("Quantized DeepSeek Model Evaluation")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Attention impl: {args.attn_implementation}")
    print(f"Dtype: {args.dtype}")
    print("=" * 80)

    compute_dtype = _torch_dtype_from_str(args.dtype)

    # Load quantized DeepSeek model
    print(f"\nLoading quantized DeepSeek model from: {args.model_path}")
    model = load_quantized_deepseek(
        args.model_path,
        attn_implementation=args.attn_implementation,
        device=args.device,
        compute_dtype=compute_dtype,
    )
    model.eval()

    # Load tokenizer saved alongside quantized model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare tasks
    task_list = [task.strip() for task in args.tasks.split(",") if task.strip()]

    # lm_eval integration
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    print("Wrapping model with HFLM...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Evaluating on tasks: {task_list}")
    for task in task_list:
        print(f"\nEvaluating task: {task}")
        print("-" * 80)
        result = evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            batch_size=args.batch_size,
        )
        if "results" in result and task in result["results"]:
            task_results = result["results"][task]
            print(f"\n{task} Results:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


