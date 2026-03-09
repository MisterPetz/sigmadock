from random import randint
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader.dataloader import Collater


def make_empty_batch() -> Batch:
    b = Batch()
    b.batch = torch.empty((0,), dtype=torch.long)
    b.edge_index = torch.empty((2, 0), dtype=torch.long)
    b.num_graphs = 0
    b.ptr = torch.tensor([0], dtype=torch.long)
    return b


class RobustCollater(Collater):
    def __call__(self, batch: Batch) -> Batch:
        valid = [d for d in batch if d is not None]
        if not valid:
            return make_empty_batch()
        try:
            return super().__call__(valid)
        except Exception as e:
            print(f"Error in collating batch: {e}")
            return make_empty_batch()


class CustomDataLoader(TorchDataLoader):
    def __init__(self, dataset: Dataset, **kwargs: dict[str, Any]) -> None:
        # Pop collate_fn to ensure it's not accidentally overridden
        kwargs.pop("collate_fn", None)

        custom_collator = RobustCollater(
            dataset=dataset,
            follow_batch=kwargs.get("follow_batch"),
            exclude_keys=kwargs.get("exclude_keys"),
        )

        # Initialize the parent DataLoader, but pass in an instance
        # of our new custom collator instead of the default one.
        super().__init__(
            dataset,
            collate_fn=custom_collator,  # We pass OUR collator here
            **kwargs,
        )


def batch_is_empty(batch: Any) -> bool:
    """Robustly detect an 'empty' batch produced by our RobustCollater."""
    if batch is None:
        return True
    # If the Batch object provides num_graphs, trust it.
    if hasattr(batch, "num_graphs"):
        try:
            return int(batch.num_graphs) == 0
        except Exception:
            # if accessing num_graphs throws, treat as non-empty to avoid wrongly skipping
            return False
    # fallback: if there is a .batch tensor (nodes -> graph mapping), empty means zero elements
    b = getattr(batch, "batch", None)
    if isinstance(b, torch.Tensor):
        return b.numel() == 0
    # Unknown object: assume non-empty (safer)
    return False


def custom_collate(data_list: list[Optional[Data]]) -> Optional[Batch]:
    """
    Filters out None values from a list of PyG Data objects and creates a batch.
    If the list is empty after filtering, it returns an empty Batch object.
    """
    print("--- CUSTOM COLLATE FUNCTION IS RUNNING ---")
    valid_data = [data for data in data_list if data is not None]

    if not valid_data:
        return Batch()  # Return an empty batch

    return Batch.from_data_list(valid_data)


def custom_collate_with_resampling(batch: Batch, dataset: Dataset, max_resample_attempts: int = 1) -> Batch:
    cleaned_batch = []
    for item in batch:
        attempts = 0
        while item is None and attempts < max_resample_attempts:
            item = dataset[randint(0, len(dataset) - 1, ()).item()]
            attempts += 1
        if item is not None:
            cleaned_batch.append(item)
    if len(cleaned_batch) == 0:
        return Batch()
    return Batch.from_data_list(cleaned_batch)
