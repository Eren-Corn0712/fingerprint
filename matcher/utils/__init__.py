import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
DEFAULT_CFG_PATH = ROOT / "matcher/cfg/default.yaml"
RANK = int(os.getenv('RANK', -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
LOGGING_NAME = 'YZUEE-Lab70636-FingerPrint-Matcher'

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for deterministic training

DEFAULT_CFG = "cfg/default.yaml"


class IterableSimpleNamespace(SimpleNamespace):
    """
    Iterable SimpleNamespace class to allow SimpleNamespace to be used with dict() and in for loops
    """

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return '\n'.join(f"{k}={v}" for k, v in vars(self).items())

    def get(self, key, default=None):
        return getattr(self, key, default)


def yaml_save(file='images.yaml', data=None):
    """
    Save YAML images to a file.
    Args:
        file (str, optional): File name. Default is 'images.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.
    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump images to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def yaml_load(file='images.yaml', append_filename=False):
    """
    Load YAML images from a file.
    Args:
        file (str, optional): File name. Default is 'images.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.
    Returns:
        dict: YAML images and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        # Add YAML filename to dict and return
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a yaml file or a yaml-formatted dictionary.
    Args:
        yaml_file: The file path of the yaml file or a yaml-formatted dictionary.
    Returns:
        None
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, default_flow_style=False)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.
    Args:
        dir_path (str) or (Path): The path to the directory.
    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    try:
        with tempfile.TemporaryFile(dir=dir_path):
            pass
        return True
    except OSError:
        return False


def emojis(string=''):
    # Return platform-dependent emoji-safe version of string
    return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m"}
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False}}})


class TryExcept(contextlib.ContextDecorator):
    # YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


# Run below code on yolo/utils init ------------------------------------------------------------------------------------

# Set logger
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging

# Check first-install steps
PREFIX = colorstr("YZU Lab70636: ")
