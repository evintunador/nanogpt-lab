from typing import List, Dict, Optional, Type, Callable, Any, Union, Tuple, Sequence
from dataclasses import dataclass
import os
import sys
import importlib

import torch
import torch.nn as nn


##################################################
#### INHERIT FROM THESE FOR TESTS/BENCHMARKS #####
##################################################


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    parallelize_plan: Dict[str, Any]
    tp_mesh_adjust_fn: Optional[Callable[[nn.Module, Any], nn.Module]] = None


@dataclass
class Competitor:
    """A competitor for testing or benchmarking."""
    module_class: Type[nn.Module]
    tp_config: Optional[TensorParallelConfig] = None
    run_filter: Callable[Union[torch.Tensor, Tuple[Any]], bool] = None


@dataclass
class ModuleTestConfig:
    """
    A dataclass to hold the full testing configuration for a module.
    """
    # A dictionary mapping competitor names to their configurations.
    competitors: Dict[str, Competitor]
    # The name of the competitor to be used as the reference for correctness checks.
    reference_competitor: str
    # A list of dictionaries, where each dict is a self-contained test case.
    test_cases: List[Dict]


@dataclass
class BenchmarkConfig:
    """
    Holds the complete benchmarking configuration for a module.
    This defines the matrix of parameters to sweep over.
    """
    # A friendly name for the module, used for filenaming.
    module_name: str
    # A dictionary mapping competitor names to their configurations.
    competitors: Dict[str, Competitor]
    # The parameter space to sweep.
    # e.g., {'dim': [1024, 2048], 'activation': ['relu', 'silu'], 'dtype': [torch.float16]}
    parameter_space: Dict[str, List[Any]]
    # A function that takes a dictionary of a single parameter combination
    # from the sweep and returns the full init_args for the module.
    # This is useful for args that are derived from others (e.g., out_dim=dim)
    init_arg_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    # A function that provides the input tensors for the module, given the init_args.
    input_provider: Callable[[Dict[str, Any], str], tuple]


##################################################
### TOOLS THE USER DOESN'T HAVE TO WORRY ABOUT ###
##################################################


def list_all_files_in_folder_and_subdirs(folder_path: str) -> List[str]:
    """
    Recursively list all .py files in the given folder and its subdirectories.
    Returns a list of file paths relative to the folder_path.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.py'):
                continue
            rel_dir = os.path.relpath(root, folder_path)
            if rel_dir == ".":
                rel_file = file
            else:
                rel_file = os.path.join(rel_dir, file)
            all_files.append(rel_file)
    return all_files


def get_available_devices(exclude: List[str] = []) -> Tuple[List[str]]:
    available_devices = []
    available_devices_with_ranks = []
    world_size = torch.cuda.device_count()

    # Always include CPU
    available_devices.append('cpu')

    # Check for CUDA devices
    if torch.cuda.is_available():
        available_devices.append('cuda')

        if world_size > 1:
            for i in range(world_size):
                available_devices_with_ranks.append(f'cuda:{i}')

    # Check for MPS devices (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')

    # Check for HIP devices (AMD ROCm)
    if hasattr(torch, 'has_hip') and torch.has_hip and torch.device('hip').type == 'hip':
        available_devices.append('hip')

        if world_size > 1:
            for i in range(world_size):
                available_devices_with_ranks.append(f'hip:{i}')

    if exclude:
        def should_keep(dev):
            return not any(ex in dev for ex in exclude)
        available_devices = [dev for dev in available_devices if should_keep(dev)]
        available_devices_with_ranks = [dev for dev in available_devices_with_ranks if should_keep(dev)]

    return available_devices, available_devices_with_ranks


def discover_dunder_objects(
        dunder: str, 
        object: Any,
        excluded_files: List[str] = ['bulk_module_test.py', 'bulk_module_benchmark.py', 'base_test_bench_utils.py']
    ) -> List[Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    all_files = list_all_files_in_folder_and_subdirs(current_dir)
    all_files = [f for f in all_files if os.path.basename(f) not in excluded_files]

    objects = []
    for file in all_files:
        try:
            relative_path = os.path.relpath(os.path.join(current_dir, file), project_root)
            module_name = relative_path.replace('.py', '').replace(os.sep, '.')
            
            module = importlib.import_module(module_name)

            obj = getattr(module, dunder, None)
            if isinstance(obj, object):
                objects.append(obj)
        except Exception as e:
            print(f"[WARNING] Could not process {file}. Error: {e}. Skipping.")

    return objects


def get_total_loss(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Computes a scalar loss from a single tensor or a tuple of tensors."""
    if isinstance(outputs, torch.Tensor):
        # Handles the common case of a single tensor output
        return outputs.sum()
    
    # Handles tuple outputs, summing only the floating point tensors
    total_loss = torch.tensor(0.0).to(outputs[0].device)
    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            total_loss += out.sum()
    return total_loss
