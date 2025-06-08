import os
import importlib
import sys
from typing import List, Dict, Any, Union, Sequence

import pytest
import torch
import torch.nn as nn

from module_test_config import ModuleTestConfig


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


def get_available_devices(exclude: List[str] = []) -> (List[str], List[str]):
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


def discover_test_configs() -> List[ModuleTestConfig]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.dirname(current_dir)

    if tests_root not in sys.path:
        sys.path.insert(0, tests_root)

    all_files = list_all_files_in_folder_and_subdirs(current_dir)
    all_files = [f for f in all_files if f not in ['bulk_module_test.py', 'module_test_config.py']]
    
    configs = []
    for file in all_files:
        try:
            relative_path = os.path.relpath(os.path.join(current_dir, file), tests_root)
            module_name = relative_path.replace('.py', '').replace(os.sep, '.')
            
            module = importlib.import_module(module_name)
            
            config = getattr(module, '__test_config__', None)
            if config:
                configs.append(config)
            else:
                print(f"[INFO] No `__test_config__` found in {file}. Skipping.")
        except Exception as e:
            print(f"[WARNING] Could not import or find `__test_config__` in {file}. Error: {e}. Skipping.")
            
    return configs


def build_test_suite(test_configs: List[ModuleTestConfig], available_devices: List[str]) -> List[Dict[str, Any]]:
    test_suite = []
    for config in test_configs:
        for i, test_case in enumerate(config.test_cases):
            # The test case may want to exclude some of the available devices
            devices_to_test = [d for d in available_devices if d not in test_case.get('excluded_devices', [])]
            for device in devices_to_test:
                test_id = f"{config.module_class.__name__}-case{i}-{device}"
                test_suite.append(
                    pytest.param(
                        config.module_class, 
                        config.kernel_class,
                        test_case, 
                        device, 
                        id=test_id
                        )
                    )
    return test_suite


def get_total_loss(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Computes a scalar loss from a single tensor or a tuple of tensors."""
    if isinstance(outputs, torch.Tensor):
        # Handles the common case of a single tensor output
        return outputs.sum()
    
    total_loss = 0
    # Handles tuple outputs, summing only the floating point tensors
    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            total_loss += out.sum()
    return total_loss


ALL_TEST_CONFIGS = discover_test_configs()
AVAILABLE_DEVICES, _ = get_available_devices()
TEST_SUITE = build_test_suite(ALL_TEST_CONFIGS, AVAILABLE_DEVICES)


@pytest.mark.parametrize("PytorchModuleCls, KernelModuleCls, test_case, device", TEST_SUITE)
def test_bulk_module(PytorchModuleCls: nn.Module, KernelModuleCls: nn.Module, test_case: Dict[str, Any], device: str):
    """
    This single function runs a complete test for one module configuration.
    Pytest calls this function repeatedly for each parameter set in TEST_SUITE.
    """
    # PyTorch Reference Implementation Test
    pytorch_module = PytorchModuleCls(**test_case['init_args']).to(device)
    pytorch_inputs = test_case['input_args'](device)
    pytorch_output = pytorch_module(*pytorch_inputs)
    test_case['output_validator'](pytorch_module, pytorch_inputs, pytorch_output)

    # Custom Kernel Implementation Test (Conditional)
    run_kernel_test = KernelModuleCls is not None
    if run_kernel_test and 'kernel_run_filter' in test_case:
        run_kernel_test = test_case['kernel_run_filter'](pytorch_inputs)
    if not run_kernel_test:
        return

    # If we proceed, run the kernel test and comparison
    kernel_module = KernelModuleCls(**test_case['init_args']).to(device)
    kernel_module.load_state_dict(pytorch_module.state_dict())
    kernel_inputs = [t.clone().detach().requires_grad_(t.requires_grad) for t in pytorch_inputs]
    kernel_output = kernel_module(*kernel_inputs)

    get_total_loss(pytorch_output).backward()
    get_total_loss(kernel_output).backward()
    tolerances = test_case.get('tolerances', {})
    if not isinstance(pytorch_output, tuple):
        pytorch_output = (pytorch_output,)
        kernel_output = (kernel_output,)

    for i, (p_out, k_out) in enumerate(zip(pytorch_output, kernel_output)):
         if isinstance(p_out, torch.Tensor):
            assert torch.allclose(p_out, k_out, **tolerances), \
                f"Forward output {i} mismatch for {PytorchModuleCls.__name__} on {device}"

    for (p_name, p_torch), (k_name, p_kernel) in zip(pytorch_module.named_parameters(), kernel_module.named_parameters()):
        assert p_torch.grad is not None and p_kernel.grad is not None, f"Gradient for {p_name} is None"
        assert torch.allclose(p_torch.grad, p_kernel.grad, **tolerances), \
            f"Gradient mismatch for parameter {p_name} on {device}"
            
    for i, (p_input, k_input) in enumerate(zip(pytorch_inputs, kernel_inputs)):
        if p_input.requires_grad and k_input.requires_grad:
            assert p_input.grad is not None and k_input.grad is not None, f"Gradient for input {i} is None"
            assert torch.allclose(p_input.grad, k_input.grad, **tolerances), \
                f"Gradient mismatch for input {i} on {device}"