from typing import List, Sequence, Union
import os
import sys
import time

from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from modules.base_test_bench_utils import ModuleBenchmarkConfig, discover_dunder_objects, get_available_devices

# Use a consistent color cycle
plot_colors = list(mcolors.TABLEAU_COLORS.keys())


def get_total_loss(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Computes a scalar loss from a single tensor or a tuple of tensors."""
    if isinstance(outputs, torch.Tensor):
        if not outputs.is_floating_point(): return None
        # Gradients need a scalar loss
        return outputs.sum()
    
    total_loss = 0
    # Handles tuple outputs, summing only the floating point tensors
    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            total_loss += out.sum()
    return total_loss

def measure_performance(
    module: torch.nn.Module, 
    inputs: tuple, 
    device: str,
    num_repeats: int = 10
) -> dict:
    """
    Measures forward/backward time and memory for a given module and input.
    Returns only the metrics that were successfully measured.
    """
    device_type = torch.device(device).type

    # Warmup
    for _ in range(5):
        outputs = module(*inputs)
        loss = get_total_loss(outputs)
        if loss is not None:
            loss.backward()
            module.zero_grad(set_to_none=True)
    
    # Synchronize before measurements
    if device_type == 'cuda': torch.cuda.synchronize()
    elif device_type == 'mps': torch.mps.synchronize()

    # --- Time Measurement ---
    fwd_time_ms = -1.0
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            outputs = module(*inputs)
        end_event.record()
        torch.cuda.synchronize()
        fwd_time_ms = start_event.elapsed_time(end_event) / num_repeats
    else:  # MPS and CPU
        start_time = time.perf_counter()
        for _ in range(num_repeats):
            outputs = module(*inputs)
        if device_type == 'mps': torch.mps.synchronize()
        end_time = time.perf_counter()
        fwd_time_ms = (end_time - start_time) * 1000 / num_repeats

    bwd_time_ms = -1.0
    loss = get_total_loss(outputs)
    if loss is not None and loss.requires_grad:
        if device_type == 'cuda':
            start_event.record()
            for _ in range(num_repeats):
                loss.backward(retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()
            bwd_time_ms = start_event.elapsed_time(end_event) / num_repeats
        else:  # MPS and CPU
            start_time = time.perf_counter()
            for _ in range(num_repeats):
                loss.backward(retain_graph=True)
            if device_type == 'mps': torch.mps.synchronize()
            end_time = time.perf_counter()
            bwd_time_ms = (end_time - start_time) * 1000 / num_repeats

    module.zero_grad(set_to_none=True)
    if device_type == 'cuda': torch.cuda.empty_cache()

    results = {
        'Forward Time (ms)': fwd_time_ms,
        'Backward Time (ms)': bwd_time_ms if bwd_time_ms > 0 else None,
    }

    # --- Memory Measurement (CUDA only) ---
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        outputs = module(*inputs)
        results['Forward Peak Memory (GB)'] = torch.cuda.max_memory_allocated(device) / 1e9

        torch.cuda.reset_peak_memory_stats(device)
        loss = get_total_loss(outputs)
        if loss is not None:
            loss.backward()
        results['Backward Peak Memory (GB)'] = torch.cuda.max_memory_allocated(device) / 1e9

        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    return {k: v for k, v in results.items() if v is not None}


def run_benchmarks(configs: List[ModuleBenchmarkConfig], device: str, output_dir: str = "modules/benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    device_type = torch.device(device).type

    for config in tqdm(configs, desc="All Benchmark Configs"):
        for plot_spec in tqdm(config.plots, desc=f"Plots for {list(config.competitors.keys())}", leave=False):
            plot_results = []
            for x_val in tqdm(plot_spec.x_vals, desc=f"Plot '{plot_spec.plot_name}' on {device}", leave=False):
                for competitor_name, competitor in config.competitors.items():
                    if competitor.module_class is None: continue
                    if device_type in (competitor.excluded_devices or []): continue
                    if competitor.tp_config: continue

                    init_args = plot_spec.init_arg_builder(x_val)
                    module = competitor.module_class(**init_args).to(device)
                    inputs = plot_spec.input_provider(init_args, device)
                    
                    perf_metrics = measure_performance(module, inputs, device)
                    
                    for metric_name, value in perf_metrics.items():
                        plot_results.append({
                            plot_spec.x_arg: x_val, 'competitor': competitor_name,
                            'measurement': metric_name, 'value': value
                        })
            
            if not plot_results: continue
            
            df = pd.DataFrame(plot_results)
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f"Benchmark: {plot_spec.plot_name} on {device_type.upper()}", fontsize=16)
            axes = axes.flatten()

            axis_map = {
                'Forward Time (ms)': (axes[0], 'Time (ms)'), 'Backward Time (ms)': (axes[1], 'Time (ms)'),
                'Forward Peak Memory (GB)': (axes[2], 'Peak Memory (GB)'), 'Backward Peak Memory (GB)': (axes[3], 'Peak Memory (GB)'),
            }
            available_metrics = df['measurement'].unique()

            for metric_name, (ax, y_label) in axis_map.items():
                if metric_name not in available_metrics:
                    ax.text(0.5, 0.5, 'Not Measured', ha='center', va='center', fontsize=12, color='gray')
                    ax.set_title(metric_name)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                
                metric_df = df[df['measurement'] == metric_name]
                ax.set_title(metric_name)
                ax.set_xlabel(plot_spec.x_arg); ax.set_ylabel(y_label)
                
                for j, (name, group) in enumerate(metric_df.groupby('competitor')):
                    ax.plot(group[plot_spec.x_arg], group['value'], marker='o', linestyle='-', label=name, color=plot_colors[j % len(plot_colors)])
                
                ax.grid(True); ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = os.path.join(output_dir, f"{plot_spec.plot_name}_{device_type}.png")
            plt.savefig(save_path)
            plt.close(fig)
            tqdm.write(f"Saved plot to {save_path}")


if __name__ == "__main__":
    benchmark_configs = discover_dunder_objects(dunder='__benchmark_config__', object=ModuleBenchmarkConfig)
    available_devices, _ = get_available_devices()
    
    if not benchmark_configs:
        print("No `__benchmark_config__` found in any module files. Nothing to do.")
        sys.exit(0)
    
    print(f"Found {len(benchmark_configs)} benchmark configurations.")
    for device in available_devices:
        if 'cpu' in device:
            continue
        print(f"--- Running benchmarks on {device} ---")
        run_benchmarks(benchmark_configs, device)