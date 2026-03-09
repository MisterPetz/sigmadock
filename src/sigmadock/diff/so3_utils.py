import sys

import numpy as np
import torch

torch_pi = torch.tensor(3.1415926535)


# hat map from vector space R^3 to Lie algebra so(3)
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    v: [..., 3]
    hat_v: [..., 3, 3]
    """
    if not torch.isfinite(v).all():
        raise ValueError(f"NaNs or infs detected in input to hat: {v}")
    if v.shape[-1] != 3:
        raise ValueError(f"Expected [..., 3] shape for hat(), got {v.shape}")
    hat_v = torch.zeros([*v.shape[:-1], 3, 3], dtype=v.dtype, device=v.device)
    hat_v[..., 0, 1], hat_v[..., 0, 2], hat_v[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return hat_v + -hat_v.transpose(-1, -2)


# vee map from Lie algebra so(3) to the vector space R^3
def vee(A: torch.Tensor) -> torch.Tensor:
    # A: [..., 3, 3]
    if not torch.allclose(A, -A.transpose(-1, -2), atol=1e-2, rtol=1e-2):
        print("Input A must be skew symmetric, Err" + str(((A - A.transpose(-1, -2)) ** 2).sum(dim=[-1, -2])))
    vee_A = torch.stack([-A[..., 1, 2], A[..., 0, 2], -A[..., 0, 1]], dim=-1)
    return vee_A


# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def Log(R: torch.Tensor) -> torch.Tensor:
    # R: [..., 3, 3]
    shape = R.shape[:-2]
    tmp_dtype = torch.float64 if R.device.type != "mps" else torch.float32
    R_ = R.reshape(-1, 3, 3).to(tmp_dtype)
    Log_R_ = rotation_vector_from_matrix(R_)
    return Log_R_.reshape([*shape, 3]).to(R.dtype)  # [..., 3]


# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R: torch.Tensor) -> torch.Tensor:
    # R: [..., 3, 3]
    return hat(Log(R))  # [..., 3, 3]


# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A: torch.Tensor) -> torch.Tensor:
    # A: [..., 3, 3]
    # MacBook MPS support
    if A.device.type == "mps":
        A_ = A.to("cpu")
        out = torch.matrix_exp(A_)  # [..., 3, 3]
        out = out.to("mps")
    else:
        out = torch.matrix_exp(A)
    return out  # [..., 3, 3]


# Exponential map from R^3 to SO(3)
def Exp(A: torch.Tensor) -> torch.Tensor:
    # A: [..., 3]
    return exp(hat(A))  # [..., 3, 3]


# Angle of rotation SO(3) to R^+, this is the norm in our chosen orthonormal basis
def Omega(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # multiplying by (1-epsilon) prevents instability of arccos when provided with -1 or 1 as input.
    # R: [..., 3, 3]
    tmp_dtype = torch.float64 if R.device.type != "mps" else torch.float32
    R_ = R.to(tmp_dtype)
    assert not torch.any(torch.abs(R) > 1.1)
    trace = torch.diagonal(R_, dim1=-2, dim2=-1).sum(dim=-1) * (1 - eps)  # [...]
    out = (trace - 1.0) / 2.0
    out = torch.clamp(out, min=-0.99, max=0.99)
    return torch.arccos(out).to(R.dtype)  # [...]


# exponential map from tangent space at R to SO(3)
def expmap(R: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    # R: [..., 3, 3]
    # tangent: [..., 3, 3]
    skew_sym = torch.einsum("...ij,...ik->...jk", R, tangent)
    if not torch.allclose(skew_sym, -skew_sym.transpose(-1, -2), atol=1e-2, rtol=1e-2):
        print("in expmap, R0.T @ tangent must be skew symmetric")
    skew_sym = (skew_sym - torch.transpose(skew_sym, -2, -1)) / 2.0  # average for numerical stability?
    exp_skew_sym = exp(skew_sym)
    return torch.einsum("...ij,...jk->...ik", R, exp_skew_sym)


# Normal sample in tangent space at R
def tangent_gaussian(R: torch.Tensor) -> torch.Tensor:
    # R0: [..., 3, 3]
    return torch.einsum(
        "...ij,...jk->...ik", R, hat(torch.randn(*R.shape[:-2], 3, dtype=R.dtype, device=R.device))
    )  # [..., 3, 3]


# sample from uniform distribution on SO(3)
# NOTE: do we still need this?
def sample_uniform(N: int, L: int = 1000) -> torch.Tensor:
    omega_grid = np.linspace(0, np.pi, L)
    cdf = np.cumsum(np.pi**-1 * (1 - np.cos(omega_grid)), 0) / (L / np.pi)
    omegas = np.interp(np.random.rand(N), cdf, omega_grid)
    axes = np.random.randn(N, 3)
    axes = omegas[..., None] * axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    axes_ = axes.reshape([-1, 3])
    Rs = exp(hat(torch.tensor(axes_)))
    Rs = Rs.reshape([N, 3, 3])
    return Rs


def check_nan(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        sys.stderr.write(
            f" NaN/Inf detected in {name}; stats: min={torch.amin(t).item()}, max={torch.amax(t).item()}\n"
        )
        sys.stderr.flush()


# New Log map adapted from geomstats
def rotation_vector_from_matrix(rot_mat: torch.Tensor) -> torch.Tensor:
    r"""Convert rotation matrix (in 3D) to rotation vector (axis-angle).

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884

    Get the angle through the trace of the rotation matrix:
    The eigenvalues are:
    :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
    so that:
    :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`
    The rotation vector is the vector associated to the skew-symmetric
    matrix
    :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

    For the edge case where the angle is close to pi,
    the rotation vector (up to sign) is derived by using the following
    equality (see the Axis-angle representation on Wikipedia):
    :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
    In nD, the rotation vector stores the :math:`n(n-1)/2` values
    of the skew-symmetric matrix representing the rotation.

    NOTE: if rot_mat=0, the function returns 0.

    Parameters
    ----------
    rot_mat : array-like, shape=[..., n, n]
        Rotation matrix.

    Returns
    -------
    regularized_rot_vec : array-like, shape=[..., 3]
        Rotation vector.
    """
    if not torch.isfinite(rot_mat).all():
        raise RuntimeError(f"NaN/Inf in rotation_vector_from_matrix inputs {rot_mat}")
    # check_nan(rot_mat, "rot_mat")
    angle = Omega(rot_mat)  # assume that rot_mat has ndim=3
    assert len(angle.shape) == 1, "cannot handle vectorized Log map here"
    # check_nan(angle, "angle")
    n_rot_mats = len(angle)
    rot_mat_transpose = torch.transpose(rot_mat, -2, -1)
    diff = rot_mat - rot_mat_transpose
    # check_nan(diff, "diff")
    rot_vec_not_pi = vee(diff)
    # check_nan(rot_vec_not_pi, "rot_vec_not_pi")

    # masking to handle cases where angles is close to 0 or pi
    mask_0 = torch.isclose(angle, torch.tensor(0.0, dtype=angle.dtype)).to(angle.dtype)
    mask_pi = torch.isclose(angle, torch_pi.to(angle.dtype), atol=1e-2).to(angle.dtype)
    mask_else = (1 - mask_0) * (1 - mask_pi)

    # use Taylor expansion for angle close to 0
    numerator = 0.5 * mask_0 + angle * mask_else
    denominator = (1 - angle**2 / 6) * mask_0 + 2 * torch.sin(angle) * mask_else + mask_pi
    rot_vec_not_pi = rot_vec_not_pi * numerator[..., None] / denominator[..., None]
    # check_nan(rot_vec_not_pi, "rot_vec_not_pi")

    # use wiki formula for angle close to pi
    vector_outer = 0.5 * (torch.eye(3, dtype=rot_mat.dtype).to(rot_mat.device) + rot_mat)  # [..., 3, 3]
    vector_outer = vector_outer + (
        torch.maximum(torch.tensor(0.0, dtype=vector_outer.dtype).to(rot_mat.device), vector_outer) - vector_outer
    ) * torch.eye(3, dtype=vector_outer.dtype).to(rot_mat.device)

    # check_nan(vector_outer, "vector_outer")
    squared_diag_comp = torch.diagonal(vector_outer, dim1=-2, dim2=-1)  # [..., 3]
    # check_nan(squared_diag_comp, "squared_diag_comp")
    diag_comp = torch.sqrt(squared_diag_comp.clamp(min=1e-6))  # [..., 3]

    norm_line = torch.linalg.norm(vector_outer, dim=-1)  # [..., 3]
    max_line_index = torch.argmax(norm_line, dim=-1)  # [...]
    selected_line = vector_outer[range(n_rot_mats), max_line_index]

    # want
    signs = torch.sign(selected_line)
    rot_vec_pi = angle[..., None] * signs * diag_comp

    rot_vec = rot_vec_not_pi + mask_pi[..., None] * rot_vec_pi
    return regularize(rot_vec)  # [..., 3]


def regularize(point: torch.Tensor) -> torch.Tensor:
    """Regularize a point to be in accordance with convention.
    In 3D, regularize the norm of the rotation vector,
    to be between 0 and pi, following the axis-angle
    representation's convention.
    If the angle is between pi and 2pi,
    the function computes its complementary in 2pi and
    inverts the direction of the rotation axis.
    Parameters

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884
    ----------
    point : array-like, shape=[...,3]
        Point.
    Returns
    -------
    regularized_point : array-like, shape=[..., 3]
        Regularized point.
    """
    theta = torch.linalg.norm(point, axis=-1)
    k = torch.floor(theta / 2.0 / torch_pi)

    # angle in [0;2pi)
    angle = theta - 2 * k * torch_pi

    # this avoids dividing by 0
    theta_eps = torch.where(torch.isclose(theta, torch.tensor(0.0, dtype=theta.dtype)), 1.0, theta)

    # angle in [0, pi]
    normalized_angle = torch.where(angle <= torch_pi, angle, 2 * torch_pi - angle)
    norm_ratio = torch.where(
        torch.isclose(theta, torch.tensor(0.0, dtype=theta.dtype)), 1.0, normalized_angle / theta_eps
    )

    # reverse sign if angle was greater than pi
    norm_ratio = torch.where(angle > torch_pi, -norm_ratio, norm_ratio)
    return torch.einsum("...,...i->...i", norm_ratio, point)
