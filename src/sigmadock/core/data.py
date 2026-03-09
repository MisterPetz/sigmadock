import logging
import random
from copy import deepcopy
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SampleCycleWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        num_samples: int = 1,
    ) -> None:
        """
        Wraps any PyTorch Dataset with a caching mechanism.

        Args:
            base_dataset (Dataset): The underlying dataset to wrap.
            num_samples (int): Number of samples to draw from each base dataset item.
        """

        self.base_dataset = base_dataset
        self.num_samples = num_samples

    def __len__(self) -> int:
        # The length here can be considered the length of the base dataset,
        # even though __getitem__ serves samples from the cache.
        return len(self.base_dataset) * self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a sample from the cache. After a fixed number of cycles through the cache,
        the cache is automatically refreshed.

        Args:
            idx: This argument is ignored because we sample from the cache.
        """
        raw_idx = idx // self.num_samples
        sample = self.base_dataset[raw_idx]
        seed = idx % self.num_samples
        if hasattr(sample, "set_seed"):
            sample.set_seed(seed)
        return sample.clone()


class CachedRecycleWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        batch_size: int,
        cache_factor: int = 2,
        num_cycles: int = 4,
        cache_strategy: Literal["random", "minimal_discrepancy"] = "random",
        dataset_len_augmentation_factor: int = 1,
    ) -> None:
        """
        Wraps any PyTorch Dataset with a caching mechanism.

        Args:
            base_dataset (Dataset): The underlying dataset to wrap.
            batch_size (int): The batch size used for training.
            cache_factor (int): How many samples to store in the cache as a factor of batch_size.
                For example, if batch_size is 32 and cache_factor is 2, the cache size will be 64.
            num_cycles (int): Number of full cache cycles (wrap-arounds) before refreshing the cache.
        """

        assert num_cycles > cache_factor, "num_cycles must be greater than cache_factor for speedup."
        self.base_dataset = base_dataset
        self.cache_size = batch_size * cache_factor
        self.num_cycles = num_cycles

        # Determines how to sample from the cache.
        self.cache_strategy = cache_strategy
        # TODO implement other caching strategies such as:
        # "single_complex_batch" and "similar_fragment_size_split"
        assert cache_strategy == "random", NotImplementedError(
            f"Cache strategy {cache_strategy} is not implemented. Only 'random' is supported."
        )

        self.cache: list[Data] = [None] * self.cache_size
        self.sample_counts: list[int] = [-1] * self.cache_size
        self.cache_ids: list[int] = [-1] * self.cache_size
        self.dataset_augmentation_factor = dataset_len_augmentation_factor

    def _populate_cache(self, raw_idx: int) -> int:
        """
        Refresh the internal cache of samples.

        Retains any slots that have never been drawn (sample_counts[idx] == 0)
        and replaces slots that have been used at least once. The old cache and
        counts are snapshotted so we never index into a cleared list.

        After this call:
        - self.cache         is a list of length self.cache_size
        - self.sample_counts is reset to all zeros

        Returns:
            idx: The index in the cache where the sample was stored.
            This index can be used to retrieve the sample from self.cache.
        """
        # Snapshot previous data
        # dataset_length = len(self.base_dataset)

        # slot = raw_idx % self.cache_size
        slot = random.randint(0, self.cache_size - 1)
        # If this slot was never sampled (count == 0), keep the old entry
        if 0 <= self.sample_counts[slot] < self.num_cycles:
            # Do not replace the sample, keep the old one
            return slot
        else:
            # Replace the sample with a new one from the base dataset
            self.cache[slot] = self.base_dataset[raw_idx]
            self.sample_counts[slot] = 0
            self.cache_ids[slot] = raw_idx
        # TODO compatibilise with "no retries -> Return None"
        return slot

    def __len__(self) -> int:
        # The length here can be considered the length of the base dataset,
        # even though __getitem__ serves samples from the cache.
        return len(self.base_dataset) * self.dataset_augmentation_factor

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a sample from the cache. After a fixed number of cycles through the cache,
        the cache is automatically refreshed.

        Args:
            idx: This argument is ignored because we sample from the cache.
        """
        # if not self.cache:
        # Re-populate the cache before anything else
        rand_idx = self._populate_cache(idx // self.dataset_augmentation_factor)
        # Randomly sample from the cache
        sample = self.cache[rand_idx]
        self.sample_counts[rand_idx] += 1
        return sample.clone()  # Return a clone to avoid in-place modifications


class IterableCachedRecycleWrapper(IterableDataset):
    def __init__(self, base_dataset: Dataset, cache_size: int, num_cycles: int, seed: int = 42) -> None:
        """An IterableDataset wrapper that:
        - shards across DDP ranks + workers
        - fills a local cache of up to `cache_size` samples at once
        - yields each cached sample `num_cycles` times, deterministically
        Args:
            base_dataset (Dataset): The underlying dataset to wrap.
            cache_size (int): How many samples to store in the cache at once.
            num_cycles (int): Number of times to yield each cached sample.
            seed (int): Random seed for reproducibility.


        """
        super().__init__()
        self.base_dataset = base_dataset
        self.cache_size = int(cache_size)
        self.num_cycles = int(num_cycles)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _get_rank_info(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __len__(self) -> int:
        total_len = len(self.base_dataset)
        rank, world_size = self._get_rank_info()
        per_rank = (total_len + world_size - 1) // world_size
        rstart = rank * per_rank
        rend = min(rstart + per_rank, total_len)
        num_for_rank = max(0, rend - rstart)
        return num_for_rank * self.num_cycles

    def __iter__(self) -> iter:
        rank, world_size = self._get_rank_info()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Deterministic shuffle seed per (epoch, rank, worker)
        global_worker_id = rank * num_workers + worker_id
        seed = self.seed + self.epoch + global_worker_id

        # Torch shuffle for indices (deterministic)
        print(f"ReSeeding rank {rank} worker {worker_id} using seed {seed}")
        g = torch.Generator()
        g.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            n = len(self.base_dataset)
        except Exception as e:
            # If base_dataset isn't sized, raise a clear error
            raise RuntimeError("base_dataset must support __len__ for this wrapper") from e

        indices = torch.randperm(n, generator=g).tolist()

        # Shard indices across ranks (simple chunking)
        per_rank = (len(indices) + world_size - 1) // world_size
        rstart = rank * per_rank
        rend = min(rstart + per_rank, len(indices))
        rank_indices = indices[rstart:rend]

        # Shard across local workers
        if worker_info:
            nw = worker_info.num_workers
            wid = worker_info.id
            per_w = (len(rank_indices) + nw - 1) // nw
            wstart = wid * per_w
            wend = min(wstart + per_w, len(rank_indices))
            worker_indices = rank_indices[wstart:wend]
        else:
            worker_indices = rank_indices

        # Use a local RNG for shuffling cache deterministically
        py_rand = random.Random(seed ^ 0x9E3779B97F4A7C15)

        it = iter(worker_indices)
        while True:
            cache = []
            # Fill cache with valid items (skip None and catch exceptions)
            while len(cache) < self.cache_size:
                try:
                    idx = next(it)
                except StopIteration:
                    break

                try:
                    item = self.base_dataset[idx]
                except Exception as e:
                    logger.warning(f"[rank {rank} w{worker_id}] error reading idx={idx}: {e}")
                    continue

                if item is None:
                    # skip invalid samples
                    continue

                # Make a defensive copy so downstream mutations don't share memory.
                # Prefer item.clone() if you know your items support an efficient clone.
                safe_item = item.clone() if hasattr(item, "clone") else deepcopy(item)
                cache.append(safe_item)

            if not cache:
                # no more data for this worker -> end iteration
                return

            # Recycle cached items num_cycles times, shuffling each cycle deterministically
            for _cycle_idx in range(self.num_cycles):
                py_rand.shuffle(cache)
                yield from cache


class IterableDeterministicRecycleWrapper(IterableDataset):
    """
    Validation wrapper that:
      - shards indices across DDP ranks and DataLoader workers deterministically
      - fills a local cache of up to `cache_size` samples
      - yields each cached sample `num_cycles` times
      - skips None / exceptions coming from `base_dataset[idx]`
      - optional deterministic shuffle (off by default for validation)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        cache_size: int = 32,
        num_cycles: int = 1,
        seed: int = 42,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.cache_size = int(cache_size)
        self.num_cycles = int(num_cycles)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)  # keep False for validation by default
        self.epoch = 0

        if not hasattr(self.base_dataset, "__len__") or not hasattr(self.base_dataset, "__getitem__"):
            raise ValueError("base_dataset must be map-style (support __len__ and __getitem__)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _get_rank_info(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __len__(self) -> int:
        total_len = len(self.base_dataset)
        rank, world_size = self._get_rank_info()
        per_rank = (total_len + world_size - 1) // world_size
        rstart = rank * per_rank
        rend = min(rstart + per_rank, total_len)
        num_for_rank = max(0, rend - rstart)
        return num_for_rank * self.num_cycles

    def __iter__(self) -> iter:  # noqa
        rank, world_size = self._get_rank_info()

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = rank * num_workers + worker_id
        seed = self.seed + self.epoch + global_worker_id

        total_len = len(self.base_dataset)
        if total_len == 0:
            return

        # deterministic indices for validation (no shuffle by default)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(total_len, generator=g).tolist()
        else:
            indices = list(range(total_len))

        # Seed other worker
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Shard indices across ranks
        per_rank = (len(indices) + world_size - 1) // world_size
        rstart = rank * per_rank
        rend = min(rstart + per_rank, len(indices))
        rank_indices = indices[rstart:rend]

        # Shard across local workers
        if worker_info is not None:
            nw = num_workers
            wid = worker_id
            per_w = (len(rank_indices) + nw - 1) // nw
            wstart = wid * per_w
            wend = min(wstart + per_w, len(rank_indices))
            worker_indices = rank_indices[wstart:wend]
        else:
            worker_indices = rank_indices

        it = iter(worker_indices)
        # local RNG used only if shuffle=True for cache shuffling
        local_rng = None
        if self.shuffle:
            # seed derived differently for cache shuffling to avoid correlation with index shuffle
            local_rng = __import__("random").Random(seed ^ 0x9E3779B97F4A7C15)

        while True:
            cache = []
            while len(cache) < self.cache_size:
                try:
                    idx = next(it)
                except StopIteration:
                    break

                try:
                    item = self.base_dataset[idx]
                except Exception as e:
                    logger.warning(f"[rank {rank} w{worker_id}] error reading idx={idx}: {e}")
                    continue

                if item is None:
                    logger.debug(f"[rank {rank} w{worker_id}] skipping None at idx={idx}")
                    continue

                if not isinstance(item, Data):
                    logger.warning(f"[rank {rank} w{worker_id}] dataset returned type {type(item)} for idx={idx}")

                safe_item = item.clone() if hasattr(item, "clone") else deepcopy(item)
                cache.append(safe_item)

            if not cache:
                return

            # deterministic cache shuffle if requested
            if self.shuffle and local_rng is not None:
                local_rng.shuffle(cache)

            for _ in range(self.num_cycles):
                yield from cache
