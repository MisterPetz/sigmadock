import torch


# TODO for future work: consider implementing Fisher scaling for rotational scores
# This is a helper function to convert raw rotational scores (d/d_theta) into equivalent
# translational units (1/A) by multiplying by each fragment's r_rms.
# This is done to ensure that the rotational scores are on the same scale as the translational scores,
# NOTE: left for completeness, not implemented in the paper.
def scale_rotational_score(
    fragment_coms: torch.Tensor,  # [M,3]
    atom_pos: torch.Tensor,  # [N,3]
    flat_frag_idx: torch.Tensor,  # [N] ∈ [0..M)
    s_R: torch.Tensor,  # [M,3] raw rotational scores (∂/∂θ)
) -> torch.Tensor:
    """
    Convert raw rotational scores (1/radian) into equivalent
    translational units (1/A) by multiplying by each fragment's r_rms.
    """
    M = s_R.size(0)
    device = s_R.device

    # Compute per fragment rms radius sum(rel^2) per fragment, then average and sqrt
    sum_sq = torch.zeros((M,), device=device)
    count = torch.zeros((M,), device=device)

    rel = atom_pos - fragment_coms[flat_frag_idx]  # [N,3]
    sum_sq = sum_sq.index_add(0, flat_frag_idx, (rel**2).sum(dim=1))
    count = count.index_add(0, flat_frag_idx, torch.ones_like(count[flat_frag_idx]))
    r_rms = torch.sqrt(sum_sq / count)  # [M]

    # Scale the rotational score
    s_R_equiv = r_rms.unsqueeze(-1) * s_R  # [M,3]

    return s_R_equiv
