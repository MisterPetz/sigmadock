import inspect
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data


# Tensorize
def tensorise_idxs(
    x: list[int] | np.ndarray[int], max_idx: int, idxs: None | list[int] | np.ndarray = None
) -> torch.Tensor:
    ret = torch.zeros(max_idx, dtype=torch.float)

    if len(x):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=torch.long)
    else:
        x = torch.tensor([], dtype=torch.long)

    if idxs is not None:
        ret[idxs] = torch.tensor(x, dtype=torch.float)
    else:
        ret[x] = 1
    return ret


def add_batch_node_attribute(batch: Batch, attr_name: str, attr_tensor: torch.Tensor) -> Batch:
    """
    Adds a node-level tensor attribute to a PyG Batch object and ensures it's
    correctly propagated to individual Data objects after to_data_list().

    Parameters:
    - batch (Batch): The batched graph data.
    - attr_name (str): Name of the attribute to add (e.g., 'ref_pos').
    - attr_tensor (Tensor): A tensor of shape [num_nodes, ...] corresponding to the attribute.
    """
    assert attr_tensor.size(0) == batch.num_nodes, "Mismatch in number of nodes"

    # Add attribute to batch
    setattr(batch, attr_name, attr_tensor)

    # Split and propagate to individual Data objects
    data_list = batch.to_data_list()
    node_counts = torch.bincount(batch.batch)
    cum_nodes = torch.cat([node_counts.new_zeros(1), node_counts.cumsum(0)])  # Same as batch.ptr lol.

    for i, data in enumerate(data_list):
        start, end = cum_nodes[i].item(), cum_nodes[i + 1].item()
        setattr(data, attr_name, attr_tensor[start:end])

    return Batch.from_data_list(data_list)


def add_batch_edge_attribute(batch: Batch, new_edges: dict[str, torch.Tensor]) -> Batch:
    """
    Append new edges (and their attributes) to a PyG Batch.

    Parameters
    ----------
    batch : Batch
        The original batched graph.
    new_edges : Dict[str, Tensor]
        Must include:
          - 'edge_index': LongTensor of shape [2, E_new]
        May also include any other edge-level tensors like:
          - 'edge_attr':    Tensor of shape [E_new, ...]
          - 'edge_entity':  LongTensor of shape [E_new]
          - ...custom features...

    Returns
    -------
    Batch
        The same `batch`, but with all edge-level attributes concatenated
        to include the new edges.
    """
    # 1) Sanity-check
    assert "edge_index" in new_edges, "You must supply 'edge_index'"
    new_ei = new_edges["edge_index"]
    assert new_ei.ndim == 2 and new_ei.size(0) == 2, f"'edge_index' must be [2, E_new], got {tuple(new_ei.shape)}"

    # E_old = batch.edge_index.size(1)
    E_new = new_ei.size(1)

    # 2) Append edge_index
    batch.edge_index = torch.cat([batch.edge_index, new_ei], dim=1)

    # 3) For every other tensor, cat on dim=0
    for name, tensor in new_edges.items():
        if name == "edge_index":
            continue

        # check length matches E_new
        assert tensor.size(0) == E_new, (
            f"Attribute '{name}' must have first dim == E_new ({E_new}), got {tensor.size(0)}"
        )

        old = getattr(batch, name, None)
        if old is None:
            # brand-new attribute: just set it
            setattr(batch, name, tensor)
        else:
            # existing: append
            setattr(batch, name, torch.cat([old, tensor], dim=0))

    return batch


def replace_batch_edge_attribute(batch: Batch, new_edges: dict[str, torch.Tensor]) -> Batch:
    """
    Replace all edge-level attributes on a PyG Batch with new ones.

    Parameters
    ----------
    batch : Batch
        The original batched graph.
    new_edges : Dict[str, Tensor]
        Must include:
          - 'edge_index': LongTensor of shape [2, E_new]
        May also include any other edge-level tensors:
          - 'edge_attr':    Tensor of shape [E_new, ...]
          - 'edge_entity':  LongTensor of shape [E_new]
          - ...custom features...

    Returns
    -------
    Batch
        The same `batch`, but with its `edge_index` and
        any provided per-edge attributes replaced by the new tensors.
    """
    # 1) Sanity-check for edge_index
    assert "edge_index" in new_edges, "You must supply 'edge_index'"
    new_ei = new_edges["edge_index"]
    assert new_ei.ndim == 2 and new_ei.size(0) == 2, f"'edge_index' must be [2, E_new], got {tuple(new_ei.shape)}"

    E_new = new_ei.size(1)

    # 2) Replace edge_index
    batch.edge_index = new_ei

    # 3) Replace or set each per-edge attribute
    for name, tensor in new_edges.items():
        if name == "edge_index":
            continue

        # ensure one entry per new edge
        assert tensor.size(0) == E_new, (
            f"Attribute '{name}' must have first dim == E_new ({E_new}), got {tensor.size(0)}"
        )

        # overwrite or set
        setattr(batch, name, tensor)

    return batch


def re_batch_with_attrs(batch: Batch, custom_attrs: list) -> Batch:
    data_list = batch.to_data_list()
    for key in custom_attrs:
        attr = getattr(batch, key)
        node_counts = torch.bincount(batch.batch)
        cum_nodes = torch.cat([node_counts.new_zeros(1), node_counts.cumsum(0)])

        for i, data in enumerate(data_list):
            start, end = cum_nodes[i].item(), cum_nodes[i + 1].item()
            setattr(data, key, attr[start:end])

    return Batch.from_data_list([Data(**d.to_dict()) for d in data_list])


def extract_init_kwargs(
    instance: Any,
    exclude: list[str] = (),
    exclude_types: list[type] = (),
    use_signature_defaults: bool = True,
) -> dict[str, Any]:
    sig = inspect.signature(instance.__class__.__init__)
    excludes: set[str] = set(exclude)
    cfg: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == "self" or param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if name in excludes:
            continue

        if hasattr(instance, name):
            val = getattr(instance, name)
            if isinstance(val, tuple(exclude_types)):
                continue
        else:
            if use_signature_defaults and param.default is not inspect.Parameter.empty:
                val = param.default
            else:
                # Skip entirely rather than inject None
                continue

        cfg[name] = val

    return cfg
