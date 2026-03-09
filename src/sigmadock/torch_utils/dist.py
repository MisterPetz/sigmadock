import os
import random

import numpy as np
import torch


def exists(val) -> bool:  # noqa: ANN001
    return val is not None


def _seed_worker(worker_id: int) -> None:
    """
    This function will be called every time a worker is spawned.
    It sets the seed for all the libraries used by PyTorch.
    """
    seed = torch.initial_seed() % 2**32  # Ensure a unique seed for each worker
    np.random.seed(seed)  # For numpy-based randomness
    random.seed(seed)  # For Python random module
    torch.manual_seed(seed)  # For PyTorch's random operations


# Distributed
def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def reliable_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.getenv("WORLD_SIZE", "1"))


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_local_size() -> int:
    return int(os.environ.get("LOCAL_SIZE", 1))


def get_global_rank() -> int:
    return get_rank() + get_local_rank()


def get_global_size() -> int:
    return get_world_size() * get_local_size()


# Slurm


def is_slurm_available() -> bool:
    return "SLURM_JOB_ID" in os.environ


def is_slurm_master() -> bool:
    return get_global_rank() == 0


def get_slurm_job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "")


def get_slurm_node_id() -> str:
    return os.environ.get("SLURM_NODEID", "")
