import os
import pickle
import time
import torch
import logging
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re
from pathlib import Path
from typing import Dict
import math
import random
import pandas as pd

from gptq import GPTQ
from modelutils import find_layers
from datautils import get_loaders
from quant.QLinear import *
from loguru import logger
from utils_alpha import compute_alpha_values

try:
    import pulp  # MILP 建模与求解器接口
except ImportError as e:
    pulp = None
    logger.warning(f"PuLP not installed. no_calib_auto_programming mode will not work. Install with: pip install pulp")

# QwenMoE attention modules (standard QKV structure)
atten_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]

# QwenMoE expert modules - 64 experts, each with gate_proj, up_proj, down_proj
expert_modules = []
for expert_idx in range(64):
    expert_modules.extend([
        f"mlp.experts.{expert_idx}.gate_proj",
        f"mlp.experts.{expert_idx}.up_proj",
        f"mlp.experts.{expert_idx}.down_proj",
    ])


def build_and_solve_global_milp(
    layer_stats_map: Dict[str, Dict[str, float]],
    candidate_bits: list,
    bpp_budget: float,
    gamma: float = 1.0,
) -> Dict[str, int]:
    """
    为全局所有线性层计算MILP最优位宽分配。
    
    参数:
        layer_stats_map: {layer_name: {"alpha": float, "variance": float}} 每个线性层的统计信息
        candidate_bits: 候选位宽列表 [2,3,4,8]
        bpp_budget: 平均位宽预算
        gamma: 形状先验指数，默认1.0
    
    返回:
        assignment: {layer_name: chosen_bit} 每个线性层分配到的位宽
    """
    if pulp is None:
        raise ImportError("PuLP is required for MILP optimization. Install with: pip install pulp")
    
    E = len(layer_stats_map)
    if E == 0:
        return {}
    
    # 将 layer_stats_map 转换为列表，保证顺序
    layer_names = sorted(layer_stats_map.keys())
    alphas = []
    variances = []
    for name in layer_names:
        stats = layer_stats_map[name]
        alpha_val = float(stats.get("alpha", float("nan")))
        variance_val = float(stats.get("variance", float("nan")))
        alphas.append(alpha_val)
        variances.append(variance_val)
    
    # 候选位宽检查
    candidate_bits = [int(b) for b in candidate_bits]
    if any(b <= 0 for b in candidate_bits):
        raise ValueError("候选位宽必须为正整数。")
    min_bit = min(candidate_bits)
    if bpp_budget < min_bit:
        raise ValueError(f"预算不可行：bpp_budget={bpp_budget} 小于最小位宽 {min_bit}。")
    
    # 计算形状先验敏感度
    alpha0 = float(pd.Series(alphas).median())
    eps = 1e-8
    sensitivities = [((alpha0 / max(a, eps)) ** gamma) for a in alphas]
    var_eps = 1e-12
    clamped_variances = [max(v, var_eps) for v in variances]
    
    # 构造代价表（无校准解析近似）
    q_b_scalar = {b: 2.0 ** (-2 * b) for b in candidate_bits}
    
    # 建立 MILP 模型
    prob = pulp.LpProblem("NoCalib_GlobalLayers", pulp.LpMinimize)
    
    # 二元决策变量 x_{l,b}
    x = {}
    for i in range(E):
        for b in candidate_bits:
            x[(i, b)] = pulp.LpVariable(f"x_{i}_{b}", lowBound=0, upBound=1, cat=pulp.LpBinary)
    
    # 目标函数：sum_l sum_b x_{l,b} * s_l * q_{l,b}
    obj_terms = []
    for i in range(E):
        s_l = sensitivities[i]
        v_l = clamped_variances[i]
        for b in candidate_bits:
            obj_terms.append(x[(i, b)] * (s_l * v_l * q_b_scalar[b]))
    prob += pulp.lpSum(obj_terms), "Total_Cost"
    
    # 约束1：每层恰好选择一个位宽
    for i in range(E):
        prob += pulp.lpSum([x[(i, b)] for b in candidate_bits]) == 1, f"one_bit_{i}"
    
    # 约束2：平均位宽预算
    prob += pulp.lpSum([x[(i, b)] * b for i in range(E) for b in candidate_bits]) <= bpp_budget * E, "bit_budget"
    
    # 求解
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    # 检查求解状态
    lp_status = pulp.LpStatus[status]
    if lp_status != "Optimal":
        raise RuntimeError(
            f"求解失败：LpStatus={lp_status}。"
            f"建议：提高 bpp_budget 或调整候选位宽范围。"
        )
    
    # 解析结果
    assignment = {}
    for i, layer_name in enumerate(layer_names):
        chosen_b = None
        for b in candidate_bits:
            if pulp.value(x[(i, b)]) >= 0.5:
                chosen_b = b
                break
        if chosen_b is None:
            raise RuntimeError(f"Layer {layer_name} 未选定位宽。")
        assignment[layer_name] = chosen_b
    
    return assignment



def get_model():
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(
        args.model, attn_implementation=args.attn_implementation, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True
    )
    print(model)
    # QwenMoE model check - use model type from config instead of isinstance
    model_type = getattr(config, 'model_type', '').lower()
    if 'qwen' not in model_type:
        logger.warning(f"Model type is {model_type}, expected qwen model")
    model.seqlen = 2048
    return model


@torch.no_grad()
def qwen_sequential(model, dataloader, dev, bit_config=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # Get number of experts from config
    num_experts = getattr(model.config, 'num_experts', 64)
    if num_experts != 64:
        logger.warning(f"Expected 64 experts, but config shows {num_experts}")
    
    # Compute alpha values if needed
    alpha_values = None
    if args.mixed_type == "mixed_with_alpha" or args.mixed_type == "no_calib_auto_programming":
        alpha_values = compute_alpha_values(model, cache_dir=args.cache_dir)
        print(f"Computed alpha values for {len(alpha_values)} layers")

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    print('Ready.')
    quantizers = {}
    
    # Track bit assignments per layer and expert for CSV export
    bit_assignments = []  # List of (layer_idx, expert_id, bit) tuples
    
    # Global bit-width assignment for no_calib_auto_programming mode
    global_bit_assignments = {}
    if args.mixed_type == "no_calib_auto_programming":
        try:
            # Parse candidate bits
            candidate_bits = [int(s.strip()) for s in args.milp_candidate_bits.split(",") if s.strip()]
            if not candidate_bits:
                raise ValueError("Empty candidate bits list")
            
            # Filter valid alpha values (exclude NaN and inf)
            valid_layer_stats = {}
            for layer_name, stats in alpha_values.items():
                if isinstance(stats, dict):
                    alpha_val = stats.get("alpha", float("nan"))
                    variance_val = stats.get("variance", float("nan"))
                else:
                    alpha_val = stats
                    variance_val = float("nan")
                if (
                    isinstance(alpha_val, (int, float))
                    and math.isfinite(alpha_val)
                    and isinstance(variance_val, (int, float))
                    and math.isfinite(variance_val)
                ):
                    valid_layer_stats[layer_name] = {
                        "alpha": float(alpha_val),
                        "variance": float(variance_val),
                    }
            
            if not valid_layer_stats:
                raise ValueError("No valid alpha/variance statistics found")
            
            print(f"\n{'='*80}")
            print(f"Running global MILP optimization for {len(valid_layer_stats)} layers...")
            print(f"Candidate bits: {candidate_bits}")
            print(f"BPP budget: {args.milp_bpp_budget}")
            print(f"Gamma: {args.milp_gamma}")
            print(f"{'='*80}\n")
            
            # Solve global MILP
            global_bit_assignments = build_and_solve_global_milp(
                layer_stats_map=valid_layer_stats,
                candidate_bits=candidate_bits,
                bpp_budget=args.milp_bpp_budget,
                gamma=args.milp_gamma
            )
            
            print(f"\n{'='*80}")
            print("Global MILP optimization completed!")
            print(f"Assigned bit-widths to {len(global_bit_assignments)} layers")
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Failed to solve global MILP: {e}")
            print(f"Global MILP failed, falling back to uniform {args.wbits}-bit quantization")
            # Fallback: use uniform quantization for all layers
            for layer_name in alpha_values.keys():
                global_bit_assignments[layer_name] = args.wbits
    
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+--------------------------------+------------+------------+------------+---------+')
        print('|              name              |weight_error| fp_inp_SNR | q_inp_SNR  |  time   |')
        print('+================================+============+============+============+=========+')

        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [list(full.keys())]
        
        # Initialize variables
        low_bit_experts = []
        high_bit_experts = []
        
        # Check if this layer has experts
        has_experts = hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts')
        
        # random generation
        if args.mixed_type == "random" and has_experts:
            import random
            numbers = list(range(num_experts))
            low_bit_config = random.sample(numbers, 2)
            for num in low_bit_config:
                numbers.remove(num)
            high_bit_config = random.sample(numbers, 2)
            low_bit_experts = [f"mlp.experts.{j}" for j in low_bit_config]   
            high_bit_experts = [f"mlp.experts.{j}" for j in high_bit_config]
        elif args.mixed_type == "manual" and has_experts:
            if bit_config is not None:
                _, indices_max = torch.topk(bit_config[i], args.h_experts)
                _, indices_min = torch.topk(bit_config[i], args.l_experts, largest=False)
                low_bit_experts = [f"mlp.experts.{j.item()}" for j in indices_min]   
                high_bit_experts = [f"mlp.experts.{j.item()}" for j in indices_max]
            else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed" and has_experts:
             if bit_config is not None:
                low_bit_experts = []
                high_bit_experts = []
                for expert_index in bit_config[i].keys():
                    if bit_config[i][expert_index] == 1:
                        low_bit_experts.append(f"mlp.experts.{expert_index}")
                    elif bit_config[i][expert_index] == 3:
                        high_bit_experts.append(f"mlp.experts.{expert_index}")
             else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed_with_alpha" and has_experts:
            # Compute alpha values for expert modules in this layer
            expert_alpha_values = {}
            layer_name_prefix = f'model.layers.{i}.mlp.experts.'
            
            # Get alpha values for all experts in this layer
            for expert_idx in range(num_experts):
                expert_prefix = f'{layer_name_prefix}{expert_idx}'
                # Get alpha values for all three weight matrices (gate_proj, up_proj, down_proj)
                alpha_sum = 0.0
                count = 0
                for weight_name in ['gate_proj', 'up_proj', 'down_proj']:
                    weight_full_name = f'{expert_prefix}.{weight_name}'
                    stats = alpha_values.get(weight_full_name)
                    if stats is not None:
                        if isinstance(stats, dict):
                            alpha_val = stats.get("alpha", float("nan"))
                        else:
                            alpha_val = stats
                        # Check if alpha is valid (not NaN and not infinite)
                        if isinstance(alpha_val, (int, float)) and math.isfinite(alpha_val):
                            alpha_sum += float(alpha_val)
                            count += 1
                
                if count > 0:
                    expert_alpha_values[expert_idx] = alpha_sum / count
                else:
                    # If no valid alpha found, use a default value
                    expert_alpha_values[expert_idx] = 0.0
            
            # Sort experts by alpha value (ascending: smaller alpha = more sensitive = higher precision)
            sorted_experts = sorted(expert_alpha_values.items(), key=lambda x: x[1])
            
            # Calculate number of experts for each precision
            total_experts = num_experts
            n_high_bit = int(total_experts * args.high_bit_experts_ratio)
            n_low_bit = int(total_experts * args.low_bit_experts_ratio)
            
            # Smaller alpha gets higher precision (wbits+1)
            high_bit_experts = [f"mlp.experts.{sorted_experts[j][0]}" for j in range(n_high_bit)]
            # Larger alpha gets lower precision (wbits-1)
            low_bit_experts = [f"mlp.experts.{sorted_experts[total_experts - j - 1][0]}" for j in range(n_low_bit)]
            
            print(f"Layer {i}: High bit experts (alpha_sorted): {[expert_alpha_values[sorted_experts[j][0]] for j in range(n_high_bit)]}")
            print(f"Layer {i}: Low bit experts (alpha_sorted): {[expert_alpha_values[sorted_experts[total_experts - j - 1][0]] for j in range(n_low_bit)]}")
        elif args.mixed_type == "no_calib_auto_programming":
            # Global MILP has already been solved, no per-layer processing needed
            pass


        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:

                gptq[name] = GPTQ(subset[name], logger, name, args.wbits)

                if args.mixed_type == "uniform":
                    # Configure quantization based on layer split option
                    if args.half_layers_expert_split:
                        # Split layers in half: first half uses wbits+1, second half uses wbits for experts
                        total_layers = len(layers)
                        if name not in expert_modules:
                            # Attention layers: keep original attn_bits
                            assigned_bit = args.attn_bits
                        else:
                            # Expert layers: first half uses wbits+1, second half uses wbits
                            if i < total_layers // 2:
                                assigned_bit = args.wbits + 1  # First half layers
                            else:
                                assigned_bit = args.wbits  # Second half layers
                    else:
                        # Default: use wbits for experts, attn_bits for attention
                        if name not in expert_modules:
                            assigned_bit = args.attn_bits
                        else:
                            assigned_bit = args.wbits
                    
                    gptq[name].quantizer.configure(assigned_bit, perchannel=True, sym=args.sym, mse=False, pack=args.pack) 
                    gptq[name].wbits = assigned_bit
                    
                    # Track bit assignment for CSV export
                    if name in expert_modules:
                        # Extract expert_id from name (e.g., "mlp.experts.0.gate_proj" -> expert_id=0)
                        name_parts = name.split('.')
                        if len(name_parts) >= 3 and name_parts[1] == 'experts' and name_parts[2].isdigit():
                            expert_id = int(name_parts[2])
                            # Only track once per expert (gate_proj, up_proj, down_proj all have same bit, so we track once)
                            if name.endswith('.gate_proj'):
                                bit_assignments.append((i, expert_id, assigned_bit))
                elif args.mixed_type == "no_calib_auto_programming":
                    # Use global MILP-assigned bit-widths
                    # Construct full layer name: model.layers.{i}.{name}
                    full_layer_name = f'model.layers.{i}.{name}'
                    
                    # Get the assigned bit-width from global assignment
                    assigned_bit = global_bit_assignments.get(full_layer_name, args.wbits)
                    gptq[name].quantizer.configure(assigned_bit, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                    gptq[name].wbits = assigned_bit
                    
                    # Track bit assignment for CSV export
                    if name in expert_modules:
                        # Extract expert_id from name (e.g., "mlp.experts.0.gate_proj" -> expert_id=0)
                        name_parts = name.split('.')
                        if len(name_parts) >= 3 and name_parts[1] == 'experts' and name_parts[2].isdigit():
                            expert_id = int(name_parts[2])
                            # Track all weight matrices (gate_proj, up_proj, down_proj) as they may have different bits in MILP mode
                            bit_assignments.append((i, expert_id, assigned_bit))
                else:
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bits
                    else:
                        assigned_bit = args.wbits
                        # Check if this expert should use high or low bit
                        # name format: "mlp.experts.{expert_id}.{weight_name}"
                        name_parts = name.split('.')
                        expert_prefix = None
                        if len(name_parts) >= 3 and name_parts[1] == 'experts' and name_parts[2].isdigit():
                            expert_id = int(name_parts[2])
                            expert_prefix = f"mlp.experts.{expert_id}"
                        
                        if expert_prefix and expert_prefix in high_bit_experts:
                            assigned_bit = args.wbits+1
                            gptq[name].quantizer.configure(assigned_bit, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = assigned_bit
                        elif expert_prefix and expert_prefix in low_bit_experts:
                            assigned_bit = args.wbits-1
                            gptq[name].quantizer.configure(assigned_bit, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = assigned_bit
                        else:
                            gptq[name].quantizer.configure(assigned_bit, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = assigned_bit
                        
                        # Track bit assignment for CSV export
                        # Extract expert_id from name (e.g., "mlp.experts.0.gate_proj" -> expert_id=0)
                        if expert_prefix:
                            expert_id = int(name_parts[2])
                            # Only track once per expert (gate_proj, up_proj, down_proj all have same bit, so we track once)
                            if name.endswith('.gate_proj'):
                                bit_assignments.append((i, expert_id, assigned_bit))
            # print(layer)
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                # quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
                quantizers['model.layers.%d.%s' % (i, name)] = None
                if args.pack:
                    # real quant for compact memory
                    quant_config = BaseQuantizeConfig(nbits=gptq[name].wbits, group_size=args.groupsize)
                    name_parts = name.split('.')
                    if len(name_parts) == 2: # atten layer
                        _module = getattr(layer, name_parts[-2])
                        linear_layer = getattr(_module, name_parts[-1])
                    else: 
                        # QwenMoE expert layer: mlp.experts.{expert_id}.{weight_name}
                        if name_parts[0] == 'mlp' and name_parts[1] == 'experts' and len(name_parts) >= 4:
                            experts = getattr(layer.mlp, "experts")
                            expert_idx = int(name_parts[2])
                            _module = experts[expert_idx]
                            linear_layer = getattr(_module, name_parts[3])
                        # QwenMoE shared expert layer:  mlp.shared_expert.gate_proj
                        elif "shared_expert" in name_parts:
                            _module = getattr(layer.mlp, "shared_expert")
                            linear_layer = getattr(_module, name_parts[-1])
                        else:
                            # Fallback for other layers
                            _module = getattr(layer, name_parts[0])
                            linear_layer = getattr(_module, name_parts[-1])
                    quant_layer = QLinear(quant_config=quant_config, device=linear_layer.weight.device)
                    quant_layer.replace_quantized_weight(linear_layer.weight, scale, zero)
                    setattr(_module, name_parts[-1], quant_layer)
                    print(getattr(_module, name_parts[-1]).W_q.dtype)
                gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print('+--------------------------------+------------+------------+------------+---------+')
        print('\n')

    model.config.use_cache = use_cache
    
    # Save bit assignments to CSV if specified
    if args.save_bit_assignments and bit_assignments:
        # Group by layer and expert (take average if multiple entries per expert)
        from collections import defaultdict
        layer_expert_bits = defaultdict(lambda: defaultdict(list))
        
        for layer_idx, expert_id, bit_value in bit_assignments:
            layer_expert_bits[layer_idx][expert_id].append(bit_value)
        
        # Calculate average bit for each layer-expert pair
        csv_data = []
        for layer_idx in sorted(layer_expert_bits.keys()):
            for expert_id in sorted(layer_expert_bits[layer_idx].keys()):
                bits = layer_expert_bits[layer_idx][expert_id]
                avg_bit = sum(bits) / len(bits)
                csv_data.append({
                    'layer': layer_idx,
                    'expert_id': expert_id,
                    'bit': round(avg_bit, 2)
                })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = args.save_bit_assignments
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved bit assignments to CSV: {csv_path}")
        print(f"Saved bit assignments for {len(csv_data)} layer-expert pairs to {csv_path}")

    return quantizers


if __name__ == "__main__":
    import argparse

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `Qwen/Qwen2.5-MoE-A14B-Chat`."
    )
    parser.add_argument(
        "--wbits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="weight bit-width",
    )
    parser.add_argument(
        "--attn_bits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="attention weight bit-width",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "gsm8k", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--gsm8k_field",
        type=str,
        choices=["question", "answer"],
        default="question",
        help="For gsm8k, choose whether to use only the question or only the answer as calibration text.",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size",
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="batch size."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument(
        '--sym', 
        action='store_true', 
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', 
        action='store_true', 
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        "--multigpu",
        action="store_true",
    )
    parser.add_argument(
        "--eval_ppl", action="store_true", help="Evaluate perplexity."
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default="",
        help="Test datasets",
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )
    parser.add_argument(
        "--use_flash_attention_2", action="store_true", help="Whether to use flash_attention2 for inference."
    )
    parser.add_argument(
        '--r', type=int, default=7, help='Number of experts to preserve'
    )
    parser.add_argument(
        "--mixed_type",
        type=str,
        choices=["uniform", "mixed", "random", "manual", "mixed_with_alpha", "no_calib_auto_programming"],
        help='Whether to use mixed-precision',
    )
    parser.add_argument(
        "--h_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--l_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--precisions", type=str, help="the file path of experts precision"
    )
    parser.add_argument(
        "--saving_path", type=str, help="the saving path of quantized model"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache alpha values"
    )
    parser.add_argument(
        "--high_bit_experts_ratio", type=float, default=0.25, help="Ratio of high bit experts (for mixed_with_alpha)"
    )
    parser.add_argument(
        "--low_bit_experts_ratio", type=float, default=0.25, help="Ratio of low bit experts (for mixed_with_alpha)"
    )
    parser.add_argument(
        "--milp_candidate_bits", type=str, default="2,3,4,8", 
        help="Candidate bit-widths for MILP (comma-separated, e.g., '2,3,4,8')"
    )
    parser.add_argument(
        "--milp_bpp_budget", type=float, default=3.5, 
        help="Average bit-width budget for MILP solver"
    )
    parser.add_argument(
        "--milp_gamma", type=float, default=1.0, 
        help="Gamma parameter for sensitivity weighting in MILP"
    )
    parser.add_argument(
        "--save_bit_assignments", type=str, default=None,
        help="Path to save layer-expert bit assignments (CSV format)"
    )
    parser.add_argument(
        "--half_layers_expert_split", action="store_true",
        help="If enabled, split layers in half: first half uses wbits+1 for experts, second half uses wbits. Attention layers keep original attn_bits."
    )

    args = parser.parse_args()
    print(f'Arguments: {args}')

    # Propagate gsm8k field selection via environment for datautils
    if args.dataset == "gsm8k":
        os.environ["GSM8K_FIELD"] = args.gsm8k_field

    groupsize = args.groupsize
    args.wbits = int(args.wbits[0])
    args.attn_bits = int(args.attn_bits[0])

    model = get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    bit_config = None
    if args.mixed_type == "manual" or args.mixed_type == "mixed":
        high_bit = args.precisions
        if os.path.exists(high_bit):
            with open(high_bit, 'rb') as file:
                bit_config = pickle.load(file)
        else:
            print("Please generate the high_experts.pkl and low_experts.pkl first!")
            exit()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    device = "cuda:0"
    tick = time.time()
    quantizers = qwen_sequential(model, dataloader, device, bit_config)
    print("quantization time:", time.time() - tick, "s")
    print(model)

    if args.eval_ppl:
        for dataset in ["wikitext2"]:#, "c4", "ptb"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=2048, model=args.model
            )
            print(dataset)
            from eval_ppl_utils import llama_eval
            t1 = time.time()
            llama_eval(model, testloader, device, dataset)
            print("Time: ", time.time() - t1)
    if args.save:
        # Calculate average bits for different mixed_type modes
        if args.mixed_type in ["manual", "mixed"] and args.precisions:
            average_bits = int(args.precisions[-9:-7])/8
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{average_bits}"
        elif args.mixed_type == "mixed_with_alpha":
            # For mixed_with_alpha, calculate average bits based on ratios
            avg_bit_ratio = (
                args.high_bit_experts_ratio * (args.wbits + 1) +
                args.low_bit_experts_ratio * (args.wbits - 1) +
                (1 - args.high_bit_experts_ratio - args.low_bit_experts_ratio) * args.wbits
            )
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{avg_bit_ratio:.2f}"
        elif args.mixed_type == "no_calib_auto_programming":
            # For MILP, use the budget as the average bits
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{args.milp_bpp_budget:.2f}"
        else:
            # For uniform or other modes, use wbits directly
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{args.wbits}"
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.save_pretrained(saving_path)
        from utils.pack import save_quantized
        save_quantized(model, saving_path)

