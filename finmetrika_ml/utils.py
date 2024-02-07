import sys
import random
import logging
import inspect
import platform
from datetime import datetime
import torch
import numpy as np



def set_all_seeds(seed:int):
    """Set the seed for all packages: python, numpy, torch, torch.cuda, and mps.

    Args:
        seed (int): Positive integer value.
    """
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda
    torch.mps.manual_seed(seed) # mps



def check_device(verbose:bool=True):
    """Check which compute device is available on the machine.

    Args:
        verbose (bool): verbose argument - prints all

    Returns:
        str: string name of the compute device available
    """
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    if verbose: 
        print(f'Using {device} device!')

    return device



def get_python_version():
    """Return the current running Python version.
    """
    return sys.version.split()[0]


def get_package_version(package_name):
    """Print the version of the Python package.

    Args:
        package_name (str): name of the package
    """
    try:
        package = __import__(package_name)
        return package.__version__
    except ImportError:
        return "Not Installed"





def update_config(FLAGS):
    """
    Update config arguments if any change was done via CLI when
    running "sh run.sh". FLAGS argument is instantiation of the
    Config dataclass.
    """
    for i in range(1, len(sys.argv),2):
        attr_name = sys.argv[i]
        attr_value = sys.argv[i+1]
    
        if hasattr(FLAGS, attr_name):
            setattr(FLAGS, attr_name, attr_value)
        else:
            logging.warning(F'No such attribute: {attr_name}')


def create_experiment_descr_file(config):
    """Create a txt file to include information on
    experiment including all the parameters used.

    Args:
        config (ProjectConfig): project parameters
    """
    # Get the parameters
    exp_description, exp_params = config.export_params()
    
    # Where to create the file
    file_path = config.dir_experiments/config.experiment_version/"experiment_info.txt"
    
    with open(file_path, "w") as f:
        f.write(f'EXPERIMENT DESCRIPTION:\n{exp_description}\
            \n\nMODEL PARAMETERS:\n')
        for key, value in exp_params.items():
            # Exclude already used information
            if key not in ["experiment_version",
                           "experiment_description"]:
                f.write(f"{key}:{value}\n")


def add_runtime_experiment_info(start_time, config):
    exp_file_path = config.dir_experiments/config.experiment_version/"experiment_info.txt"
    end_time = datetime.now()
    dtime = end_time-start_time
    with open(exp_file_path, "a") as f:
        f.write(f'\nTIME END: {end_time.strftime("%Y-%m-%d %H:%M:%S")}\nRUN TIME: {dtime}')
        f.write(f'\n\nPython Version: {get_python_version()}')
        f.write(f'\nOS: {platform.system()} {platform.release()}')
        f.write(f'\nTorch Version: {get_package_version("torch")}')
        f.write(f'\nTransformers Version: {get_package_version("transformers")}')
        f.write(f'\nNumPy Version: {get_package_version("numpy")}')
        f.write(f'\nPandas Version: {get_package_version("pandas")}')



# Functions for documentation
def generate_markdown_doc(func):
    # Extract function signature
    sig = inspect.signature(func)
    func_name = func.__name__
    docstring = inspect.getdoc(func) or ''
    
    # Prepare Markdown for the function signature
    args_str = ', '.join([f"{p.name}: {p.annotation.__name__ if hasattr(p.annotation, '__name__') else 'Any'}" 
                          for p in sig.parameters.values()])
    markdown_output = f"### `{func_name}` {{.unnumbered}}\n> {func_name}({args_str})\n\n"
    
    # Function description (first paragraph of the docstring)
    func_description = docstring.split('\n\n')[0] if docstring else ''
    markdown_output += f"*{func_description}*\n\nArguments:\n\n"
    
    # Table header
    markdown_output += "|       | type    |default| description|\n|--------|--------|--------|--------|\n"
    
    # Extract parameter descriptions
    param_descriptions = {}
    if "Args:" in docstring:
        args_index = docstring.index("Args:")
        args_section = docstring[args_index:]
        for line in args_section.split('\n')[1:]:  # skip the "Args:" line
            line = line.strip()
            if line:  # non-empty line
                parts = line.split(':')
                if len(parts) > 1:
                    param_name = parts[0].split(' ')[0]  # assuming "param_name (type)" format
                    description = ':'.join(parts[1:]).strip()
                    param_descriptions[param_name] = description
    
    # Parse parameters and defaults from signature
    for param in sig.parameters.values():
        param_name = param.name
        param_type = param.annotation.__name__ if hasattr(param.annotation, '__name__') else 'Any'
        default = param.default if param.default != inspect.Parameter.empty else 'None'
        description = param_descriptions.get(param_name, "Description not available")
        
        markdown_output += f"| **{param_name}** | {param_type} | {default} | {description} |\n"
    
    return markdown_output

