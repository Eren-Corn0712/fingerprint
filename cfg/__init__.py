es (62 sloc)  2.67 KB

import contextlib
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from utils import LOGGER, colorstr, yaml_load, DEFAULT_CFG, IterableSimpleNamespace

CLI_HELP_MSG = "Pl"


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary.
    This function converts a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.
    Inputs:
        cfg (str) or (Path) or (SimpleNamespace): Configuration object to be converted to a dictionary.
    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.
    Args:
        cfg (str) or (Path) or (Dict) or (SimpleNamespace): Configuration data.
        overrides (str) or (Dict), optional: Overrides in the form of a file name or a dictionary. Default is None.
    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_cfg_mismatch(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Type checks
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg_mismatch(base: Dict, custom: Dict):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.
    Inputs:
        - custom (Dict): a dictionary of custom configuration options
        - base (Dict): a dictionary of base configuration options
    """
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    if mismatched:
        for x in mismatched:
            matches = get_close_matches(x, base, 3, 0.6)
            match_str = f"Similar arguments are {matches}." if matches else 'There are no similar arguments.'
            LOGGER.warning(f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}")
        LOGGER.warning(CLI_HELP_MSG)
        sys.exit()