import importlib
import os

import torch
import torch.distributed as dist

###########################################################
###################### DYNAMIC IMPORTING ##################
###########################################################

def import_from_nested_path(nested_folders, filename, items):
    """
    Dynamically import specific items from a nested module path.
    
    Args:
        nested_folders (list): List of folder names forming the module path.
        filename (str): The file/module name without extension.
        items (list): List of item names to import from the module.
        
    Returns:
        dict: Dictionary containing the successfully imported items.
        
    Example:
        >>> imported_objects = import_from_nested_path(['tokenizers'], 'BPE', ['get_tokenizer'])
        >>> get_tokenizer = imported_objects.get('get_tokenizer')
    """
    try:
        # Construct the module path from a list of folders
        module_path = ".".join(nested_folders) + "." + filename
        
        # Dynamically import the module
        module = importlib.import_module(module_path)
        
        # Extract specific items (functions, classes, etc.)
        imported_items = {}
        for item in items:
            if hasattr(module, item):
                imported_items[item] = getattr(module, item)
            else:
                print(f"{item} is not available in {module_path}")
        return imported_items
                
    except ImportError as e:
       raise e


# a wrapper to force a given function to behave using a specified working directory rather than the current working directory
def run_in_directory(func, path, *args, **kwargs):
    """
    Execute a function in a specified working directory.
    
    Args:
        func (callable): The function to execute.
        path (str): The directory path to temporarily change to.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Any: The return value of the executed function.
        
    Example:
        >>> def example_function():
        ...     print("Current Working Directory:", os.getcwd())
        >>> run_in_directory(example_function, "models/modules/")
    """
    original_dir = os.getcwd()  # Save the current directory
    os.chdir(path)  # Change to the target directory
    try:
        result = func(*args, **kwargs)  # Execute the function
    finally:
        os.chdir(original_dir)  # Change back to the original directory
    return result


###########################################################
########### Serialization/Deserialization ##################
###########################################################


YamlJsonSafeType = Union[str, int, float, bool, None, List["YamlJsonSafeType"], Dict[str, "YamlJsonSafeType"]]

def yaml_dump(data: YamlJsonSafeType, file: TextIO, **kwargs) -> None:
    """
    Dump data to a YAML file.

    Args:
        data (YamlJsonSafeType): The data to dump.
        file (TextIO): The file object to write to.
        **kwargs: Additional keyword arguments to pass to the `yaml.dump` function.

    Example:
    """


###########################################################
###################### 4D PARALLELISM ##################
###########################################################

class ProcessGroupManager:
    """from https://github.com/huggingface/picotron/blob/main/picotron/process_group_manager.py"""
    def __init__(self, tp_size, cp_size, pp_size, dp_size):
        assert all(x == 1 for x in (tp_size, cp_size, pp_size)), f"tp, cp and pp are not yet supported"
            # TODO remove this once they are

        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))
        
        assert self.world_size == tp_size * cp_size * pp_size * dp_size, f"World size ({self.world_size}) != TP ({tp_size}) * CP ({cp_size}) * PP ({pp_size}) * DP ({dp_size})"

        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, cp_size, tp_size)  # DP * PP * CP * TP grid
        # Find the position of the current process in the grid
        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()

        # Process group creation - Update indexing to match new grid order
        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, c, :].tolist() for d in range(dp_size) for p in range(pp_size) for c in range(cp_size)])[0]
        self.cp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, :, t].tolist() for d in range(dp_size) for p in range(pp_size) for t in range(tp_size)])[0]
        self.pp_group = dist.new_subgroups_by_enumeration([self.grid[d, :, c, t].tolist() for d in range(dp_size) for c in range(cp_size) for t in range(tp_size)])[0]
        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, c, t].tolist() for p in range(pp_size) for c in range(cp_size) for t in range(tp_size)])[0]
        self.cp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, :, t].flatten().tolist() for p in range(pp_size) for t in range(tp_size)])[0]
        self.pp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, :, c, t].flatten().tolist() for c in range(cp_size) for t in range(tp_size)])[0]

        self.world_group = dist.group.WORLD
        
        # Update group IDs with new grid ordering
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, self.cp_rank, :].tolist()
        self.cp_group_ids = self.grid[self.dp_rank, self.pp_rank, :, self.tp_rank].tolist()
        self.pp_group_ids = self.grid[self.dp_rank, :, self.cp_rank, self.tp_rank].tolist()
        self.dp_group_ids = self.grid[:, self.pp_rank, self.cp_rank, self.tp_rank].tolist()
        self.cp_dp_group_ids = self.grid[:, self.pp_rank, :, self.tp_rank].flatten().tolist()
               
        # Tensor parallelism
        self.tp_world_size = dist.get_world_size(group=self.tp_group)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]
        
        # Context parallelism
        self.cp_world_size = dist.get_world_size(group=self.cp_group)
        self.cp_first_rank = self.cp_group_ids[0]
        self.cp_last_rank = self.cp_group_ids[-1]
        self.cp_send_rank = self.cp_group_ids[(self.cp_rank + 1) % self.cp_world_size]
        self.cp_recv_rank = self.cp_group_ids[(self.cp_rank - 1) % self.cp_world_size]

        # Pipeline parallelism
        self.pp_world_size = dist.get_world_size(group=self.pp_group)
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else int(self.grid[self.dp_rank, self.pp_rank + 1, self.cp_rank, self.tp_rank].item())
        self.pp_prev_rank = None if self.pp_rank == 0 else int(self.grid[self.dp_rank, self.pp_rank - 1, self.cp_rank, self.tp_rank].item())

        # Data parallelism
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]
        
        # Context + Data paralellism
        self.cp_dp_world_size = dist.get_world_size(group=self.cp_dp_group)
        
    def __str__(self):
        return f"TP({self.tp_world_size})-CP({self.cp_world_size})-PP({self.pp_world_size})-DP({self.dp_world_size})-Rank({self.global_rank})"

def setup_process_group_manager(tp_size, cp_size, pp_size, dp_size):
    global process_group_manager
    process_group_manager = ProcessGroupManager(tp_size, cp_size, pp_size, dp_size)

###########################################################
############ PRETTY/SMART VISUALIZAITONS ##################
###########################################################

def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # demo the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")