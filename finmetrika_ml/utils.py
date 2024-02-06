import sys
import random
import logging
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


def check_device(verbose:bool):
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
