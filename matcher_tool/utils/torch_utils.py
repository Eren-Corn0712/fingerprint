import math
import os
import platform
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import thop
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from matcher_tool.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER
from matcher_tool import __version__
from matcher_tool.utils.check import check_version

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_1_11 = check_version(torch.__version__, '1.11.0')
TORCH_1_12 = check_version(torch.__version__, '1.12.0')


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])


def smart_inference_mode():
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


def DDP_model(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if TORCH_1_11:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def select_device(device='', batch=0, newline=False):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'🌸Yuan Ze University Finger Print Matcher{__version__}🌸 ' \
        f'Python-{platform.python_version()} torch-{torch.__version__}'
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no ' \
                      'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                             f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                             f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                             f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                             f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                             f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                             f'{install}')

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and TORCH_1_12:  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

class FeatureExtractor:
    def __init__(self, model, layers: List[str]):
        self.model = model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}
        self.hooks = []

    def get_hooks(self):
        # You can modify this part, it's dependent on your model.
        for layer_id in self.layers:
            layer = self.model._modules[layer_id]
            self.hooks.append(layer.register_forward_hook(self.save_outputs_hook(layer_id)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.features[layer_id] = output

        return fn
