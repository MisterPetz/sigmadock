import torch
from numpy import array, float32, ndarray, pi
from rdkit import Chem
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch


def get_random_rotation_matrix(device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generates a random 3x3 SO(3) rotation matrix.

    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """
    # 1. Generate a random rotation object
    random_rotation = Rotation.random()

    # 2. Convert it to a 3x3 matrix (as a NumPy array)
    rotation_matrix_np = random_rotation.as_matrix()

    # 3. Convert to a PyTorch tensor with the correct device and dtype
    rotation_matrix_torch = torch.from_numpy(rotation_matrix_np).to(device=device, dtype=dtype)

    return rotation_matrix_torch


def get_coordinates(mol: Chem.Mol | Chem.Conformer, heavy_only: bool = False) -> ndarray:
    """Return (N,3) NumPy array of heavy atom (x, y, z) coordinates."""
    conf = mol.GetConformer() if isinstance(mol, Chem.Mol) else mol
    if heavy_only:
        coords = array(
            [conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1], dtype=float32
        )
    else:
        coords = array([conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()], dtype=float32)
    return coords


def frag2coords(batch: Batch) -> torch.Tensor:
    """Converts local fragment coordinates to global coordinates."""
    trans = torch.repeat_interleave(batch.trans, batch.num_coords_per_frag, dim=0)
    R = torch.repeat_interleave(batch.R, batch.num_coords_per_frag, dim=0)
    global_coords = torch.matmul(R, batch.local_coords[..., None]).squeeze(-1) + trans
    return global_coords  # [n, 3]


def get_fourier_embeddings(positions: torch.Tensor, sigma: float, num_features: int) -> torch.Tensor:
    """
    Computes Fourier positional embeddings for a given set of positions.
    Adapted from https://github.com/jmclong/random-fourier-features-pytorch/tree/main.

    Args:
        positions (torch.Tensor): A tensor of shape (N,) representing the position of each atom.
        sigma (float): A scaling factor for the Fourier coefficients.
        num_features (int): The number of frequencies to compute.

    Returns:
        torch.Tensor: A tensor of shape (N, 2 * num_features) containing the concatenated
                      sine and cosine Fourier embeddings for each position.
    """

    j = torch.arange(num_features, device=positions.device)
    coeffs = 2 * pi * sigma ** (j / num_features)
    embeddings = coeffs * torch.unsqueeze(positions, -1)
    embeddings_cat = torch.cat((torch.cos(embeddings), torch.sin(embeddings)), dim=-1)
    return embeddings_cat.flatten(-1)
