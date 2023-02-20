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
