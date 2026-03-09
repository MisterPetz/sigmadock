import torch


def std_from_reference(
    T_0: list[torch.Tensor], refs: torch.Tensor | list[torch.Tensor] | None = None, unbiased: bool = True
) -> torch.Tensor:
    """
    Compute the std-dev of fragment coords in T_0 relative to a reference coordinate.

    Args:
      T_0: list of [n_i, D] tensors of fragment coordinates per ligand.
      refs: torch.Tensor [D] for a global ref, or list of [D] tensors per ligand, or None (defaults to origin).
      unbiased: whether to use unbiased variance (divide by N-1) or biased (divide by N).

    Returns:
      torch.Tensor [D]: the coordinate-wise standard deviation of (coords ref).
    """
    # Prepare per-ligand reference list
    if refs is None:
        refs_list = [torch.zeros(coords.size(1), device=coords.device) for coords in T_0]
    elif isinstance(refs, torch.Tensor):
        refs_list = [refs for _ in T_0]
    else:
        if len(refs) != len(T_0):
            raise ValueError("len(refs) must match len(T_0)")
        refs_list = refs  # type: ignore[list-item]

    # Compute diffs and concatenate
    diffs = []
    for coords, ref in zip(T_0, refs_list):
        diffs.append(coords - ref.unsqueeze(0))  # [n_i, D]

    all_diffs = torch.cat(diffs, dim=0)  # [total_fragments, D]
    return all_diffs.std(dim=0, unbiased=unbiased)
