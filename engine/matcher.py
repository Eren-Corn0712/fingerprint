import os
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from utils import (DEFAULT_CFG, LOGGER, RANK, TQDM_BAR_FORMAT, colorstr, emojis,
                   yaml_save)
from utils.files import increment_path
from cfg import get_cfg


class BaseMatcher(object):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        self.args = get_cfg(cfg, overrides)
        self.console = LOGGER

        # Dirs
        project = self.args.project
        name = self.args.name or f"{self.args.mode}"
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in {-1, 0} else True))

        if RANK in {-1, 0}:
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
