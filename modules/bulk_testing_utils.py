import os
from typing import List, Dict, Callable, Optional, Type
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class ComponentTestConfig:
    """
    A dataclass to hold the full testing configuration for a component.
    """
    # A list of dictionaries, where each dict is a self-contained test case.
    test_cases: List[Dict]
    # The pure PyTorch nn.Module class.
    module_class: Type[nn.Module]
    # The optional kernel-enabled nn.Module class, which must inherit from module_class.
    kernel_module_class: Optional[Type[nn.Module]] = None


def list_all_files_in_folder_and_subdirs(folder_path):
    """
    Recursively list all files in the given folder and its subdirectories.
    Returns a list of file paths relative to the folder_path.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            rel_dir = os.path.relpath(root, folder_path)
            if rel_dir == ".":
                rel_file = file
            else:
                rel_file = os.path.join(rel_dir, file)
            all_files.append(rel_file)
    return all_files


def get_available_devices() -> List[str]:
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
                
    return available_devices, available_devices_with_ranks