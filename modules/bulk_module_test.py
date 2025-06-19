import os
import importlib
import sys
from typing import List, Dict, Any, Union, Sequence

import pytest
import torch
import torch.nn as nn

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    get_available_devices,
    discover_dunder_objects,
)


def build_test_suite(test_configs: List[ModuleTestConfig], available_devices: List[str]) -> List[Any]:
    test_suite = []
    
    for config in test_configs:
        
        # Get the reference competitor class
        ref_competitor_config = config.competitors.get(config.reference_competitor)
        if not ref_competitor_config or not ref_competitor_config.module_class:
            print(f"[WARNING] Reference competitor '{config.reference_competitor}' not found or has no module_class. Skipping.")
            continue

        ReferenceModuleCls = ref_competitor_config.module_class

        # Compare every other competitor to the reference
        for competitor_name, competitor_config in config.competitors.items():
            if competitor_name == config.reference_competitor:
                continue

            # For now, we only test non-TP modules in this suite.
            if competitor_config.tp_config:
                continue
                
            CompetitorModuleCls = competitor_config.module_class
            if CompetitorModuleCls is None:
                continue

            for i, test_case in enumerate(config.test_cases):
                competitor_exclusions = competitor_config.excluded_devices or []
                devices_to_test = [d for d in available_devices if d not in competitor_exclusions]
                
                for device in devices_to_test:
                    test_id = f"{config.reference_competitor}_vs_{competitor_name}-case{i}-{device}"
                    test_suite.append(
                        pytest.param(
                            ReferenceModuleCls,
                            CompetitorModuleCls,
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


ALL_TEST_CONFIGS = discover_dunder_objects(dunder='__test_config__', object=ModuleTestConfig)
AVAILABLE_DEVICES, _ = get_available_devices()
TEST_SUITE = build_test_suite(ALL_TEST_CONFIGS, AVAILABLE_DEVICES)

# Add this to make pytest show more info about parameterized tests
if len(TEST_SUITE) == 0:
    print("[ERROR] No tests generated!")
    exit()


@pytest.mark.parametrize("ReferenceModuleCls, CompetitorModuleCls, test_case, device", TEST_SUITE)
def test_bulk_module_correctness(
    ReferenceModuleCls: nn.Module, 
    CompetitorModuleCls: nn.Module, 
    test_case: Dict[str, Any], 
    device: str
):
    """
    This function tests that a 'competitor' module implementation is numerically equivalent
    to a 'reference' implementation (e.g., a kernel vs. pure PyTorch).
    Pytest calls this function repeatedly for each parameter set in TEST_SUITE.
    """
    
    # Handle the dummy test case
    if ReferenceModuleCls is None:
        pytest.fail("No tests were generated. Check the debug output above.")
    
    # Instantiate and validate the reference module
    ref_module = ReferenceModuleCls(**test_case['init_args']).to(device)
    ref_inputs = test_case['input_args'](device)
    
    if not all(isinstance(t, torch.Tensor) for t in ref_inputs):
        raise TypeError("All inputs provided by 'input_args' must be torch.Tensors.")
    
    # Run a validator on the reference implementation output to catch baseline bugs
    ref_outputs = ref_module(*ref_inputs)
    if 'pytorch_output_validator' in test_case:
        outputs_for_validator = ref_outputs if isinstance(ref_outputs, tuple) else (ref_outputs,)
        test_case['pytorch_output_validator'](ref_module, ref_inputs, outputs_for_validator)

    # Check if the competitor module should be run
    if 'kernel_run_filter' in test_case and not test_case['kernel_run_filter'](ref_inputs):
        pytest.skip(f"Skipping {CompetitorModuleCls.__name__} on {device} due to kernel_run_filter.")
        return

    # Instantiate the competitor module and copy weights
    competitor_module = CompetitorModuleCls(**test_case['init_args']).to(device)
    competitor_module.load_state_dict(ref_module.state_dict())
    
    competitor_inputs = [t.clone().detach().requires_grad_(t.requires_grad) for t in ref_inputs]
    competitor_outputs = competitor_module(*competitor_inputs)

    # Test numerical equivalence
    get_total_loss(ref_outputs).backward()
    get_total_loss(competitor_outputs).backward()
    
    tolerances = test_case.get('tolerances', {})
    
    ref_outputs_tuple = ref_outputs if isinstance(ref_outputs, tuple) else (ref_outputs,)
    competitor_outputs_tuple = competitor_outputs if isinstance(competitor_outputs, tuple) else (competitor_outputs,)

    for i, (ref_out, comp_out) in enumerate(zip(ref_outputs_tuple, competitor_outputs_tuple)):
         if isinstance(ref_out, torch.Tensor):
            assert torch.allclose(ref_out, comp_out, **tolerances), \
                f"Forward output {i} mismatch for {CompetitorModuleCls.__name__} vs {ReferenceModuleCls.__name__} on {device}"

    for (p_name, p_ref), (_, p_comp) in zip(ref_module.named_parameters(), competitor_module.named_parameters()):
        assert p_ref.grad is not None and p_comp.grad is not None, f"Gradient for {p_name} is None in one of the modules"
        assert torch.allclose(p_ref.grad, p_comp.grad, **tolerances), \
            f"Gradient mismatch for parameter {p_name} on {device}"
            
    for i, (ref_in, comp_in) in enumerate(zip(ref_inputs, competitor_inputs)):
        if ref_in.requires_grad and comp_in.requires_grad:
            assert ref_in.grad is not None and comp_in.grad is not None, f"Gradient for input {i} is None in one of the modules"
            assert torch.allclose(ref_in.grad, comp_in.grad, **tolerances), \
                f"Gradient mismatch for input {i} on {device}"