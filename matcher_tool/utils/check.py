# Ultralytics YOLO 🚀, GPL-3.0 license
import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import urllib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pkg_resources as pkg
import psutil
import torch
from matplotlib import font_manager

from matcher_tool.utils import (AUTOINSTALL, LOGGER, ROOT, TryExcept, colorstr, emojis)


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

def check_version(current: str = '0.0.0',
                  minimum: str = '0.0.0',
                  name: str = 'version ',
                  pinned: bool = False,
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required minimum version.
    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.
    Returns:
        bool: True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f'WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(warning_message)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result