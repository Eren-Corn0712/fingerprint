import os
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

from matcher_tool.utils import (DEFAULT_CFG, LOGGER, RANK, TQDM_BAR_FORMAT, colorstr, emojis,
                                yaml_save, Profile)
from matcher_tool.utils.check import print_args
from matcher_tool.utils.files import increment_path
from matcher_tool.cfg import get_cfg
from matcher_tool.data.dataset import FingerPrintDataset


class BaseMatcher(object):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        self.args = get_cfg(cfg, overrides)
        self.console = LOGGER
        self.profile = Profile
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

        if RANK == -1:
            print_args(vars(self.args))

        self.data = self.args.data
        self.dataset = self.get_dataset()
        self.csv = self.save_dir / 'results.csv'

    def log(self, text, rank=-1):
        """
        Logs the given text to given ranks process if provided, otherwise logs to all ranks.
        Args"
            text (str): text to log
            rank (List[Int]): process rank
        """
        if rank in {-1, 0}:
            self.console.info(text)

    def warning(self, text, rank=-1):
        if rank in {-1, 0}:
            self.console.warning(text)

    def preprocess_image(self, labels):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """

        return labels

    def get_dataset(self):
        # TODO: We will support different type dataset.
        return FingerPrintDataset(self.data)
