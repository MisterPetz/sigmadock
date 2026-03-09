import logging

import torch

def random_unit_vectors_resample(B: int, dim: int = 3, min_norm: float = 1e-3, device=None, dtype=None):
    """
    Samples until every vector has Euclidean norm >= min_norm, then normalizes.
    """
    v = torch.rand(B, dim, device=device, dtype=dtype)
    norm = v.norm(dim=1, keepdim=True)
    bad = norm < min_norm

    # Keep resampling just the “bad” ones
    while bad.any():
        count_bad = bad.sum().item()
        v[bad] = torch.rand(count_bad, dim, device=device, dtype=dtype)
        norm = v.norm(dim=1, keepdim=True)
        bad = norm < min_norm

    return v / norm

def init_edge_rot_mat(edge_distance_vec: torch.Tensor, eps: float = 1e-5, verbose: bool = False) -> torch.Tensor:
    """
    Original-style rotation init, but with safe fallbacks for zero/near-zero edges.
    Args:
        edge_distance_vec: Tensor of shape [B, 3]
        eps: Minimum norm below which we substitute a default direction.
    Returns:
        edge_rot_mat: Tensor of shape [B, 3, 3], detached.
    """
    B = edge_distance_vec.size(0)
    device = edge_distance_vec.device
    edge_vec_0 = edge_distance_vec

    # 1) Compute norms & mask
    edge_vec_0_dist = torch.norm(edge_vec_0, dim=1)                  # [B]
    mask_good = edge_vec_0_dist > eps                                # [B]
    if not mask_good.all():
        if verbose:
            print(f"edge_vec_0_dist: {edge_vec_0_dist}")
            logging.warning(f"Found {(~mask_good).sum().item()} near-zero edge(s); using default direction")

    # 2) Normalize safely (clamp length then divide)
    safe_dist = edge_vec_0_dist.clamp(min=eps).view(-1, 1)            # [B,1]
    norm_x = edge_vec_0 / safe_dist                                  # [B,3]
    # Re-normalize to unit length (numerically stable)
    norm_x = norm_x / torch.norm(norm_x, dim=1, keepdim=True)  # [B,3]
    
    # 3) Default-fallback for bad entries
    default_dir = torch.tensor([1.0, 0.0, 0.0], device=device)       # x-axis
    norm_x = torch.where(mask_good.view(-1, 1), norm_x, default_dir)

    # 4) Copy original algorithm, using norm_x with both good & fallback entries
    edge_vec_2 = 2 * (torch.rand_like(edge_vec_0) - 0.5)
    edge_vec_2 = edge_vec_2 / torch.norm(edge_vec_2, dim=1, keepdim=True).clamp(min=eps)
    # Re-normalize to unit length (numerically stable)
    edge_vec_2 = edge_vec_2 / torch.norm(edge_vec_2, dim=1, keepdim=True)  # [B,3]
    
    # Two 90° rotations of edge_vec_2
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0], edge_vec_2b[:, 1] = -edge_vec_2[:, 1], edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1], edge_vec_2c[:, 2] = -edge_vec_2[:, 2], edge_vec_2[:, 1]

    # Pick least-aligned
    def absdot(v): return torch.abs((v * norm_x).sum(dim=1, keepdim=True))
    dots = torch.cat([absdot(edge_vec_2), absdot(edge_vec_2b), absdot(edge_vec_2c)], dim=1)  # [B,3]
    best = dots.argmin(dim=1)                                                             # [B]
    candidates = torch.stack([edge_vec_2, edge_vec_2b, edge_vec_2c], dim=1)               # [B,3,3]
    edge_vec_2 = candidates[torch.arange(B), best]                                        # [B,3]

    # 5) Build orthonormal frame via cross products (with safe norms)
    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / torch.norm(norm_z, dim=1, keepdim=True).clamp(min=eps)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.norm(norm_y, dim=1, keepdim=True).clamp(min=eps)

    # 6) Assemble rotation matrices
    nx = norm_x.view(B, 3, 1)
    ny = (-norm_y).view(B, 3, 1)   # negate y for right-handed
    nz = norm_z.view(B, 3, 1)
    rot_inv = torch.cat([nz, nx, ny], dim=2)  # [B,3,3]
    rot_mat = rot_inv.transpose(1, 2)         # [B,3,3]

    return rot_mat.detach()

