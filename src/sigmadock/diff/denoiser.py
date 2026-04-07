from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from sigmadock.chem.processing import get_lig_idxs, get_local_interactions_batched
from sigmadock.diff import so3_utils
from sigmadock.diff.se3_diffuser import SE3Diffuser
from sigmadock.diff.so3_diffuser import d_log_f_d_omega
from sigmadock.oracle import HPARAMS
from sigmadock.torch_utils.utils import (
    replace_batch_edge_attribute,
)  # add_batch_node_attribute,


class SigmaDockDenoiser(nn.Module):
    """
    SigmaDockDenoiser is a class that represents a denoising model for SigmaDock used for training and inference.
    """

    def __init__(
        self,
        model: nn.Module,
        # Graph Parameters
        include_interactions: bool = True,
        # r3 parameters
        min_beta: float = 0.1,
        max_beta: float = 20.0,
        # so3 parameters
        schedule: str = "logarithmic",
        min_sigma: float = 0.1,
        max_sigma: float = 1.5,
        num_sigma: int = 1000,
        num_omega: int = 2000,
        cache_path: str | Path = Path("cache"),
        use_cached_score: bool = True,
        L: int = 1000,
        # Rot Score Options
        rot_score_method: Literal[
            "space", "score"
        ] = "score",  # Either predict space: delta_R or score_R from the model
        rot_score_scaling: Literal["rms", "true"] | None = "rms",  # How to scale the rotation score
        reverse_rotations: bool = False,  # Deprecated compatibility.
        # Cutoffs for local dynamic edges
        cutoff_complex_interactions: float = -1,
        cutoff_fragment_interactions: float = -1,
        cutoff_complex_virtual: float = -1,
        # Misc
        verbose: bool = False,
        # **kwargs are ignored but can be used for future extensions
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SigmaDockDenoiser with a given model.
        Args:
            model (nn.Module): The model to be used for denoising.
            include_interactions (bool): Flag to include interactions in the model.
            min_beta (float): Minimum beta value for R3 diffusion.
            max_beta (float): Maximum beta value for R3 diffusion.
            schedule (str): Schedule for SO3 diffusion.
            min_sigma (float): Minimum sigma value for SO3 diffusion.
            max_sigma (float): Maximum sigma value for SO3 diffusion.
            num_sigma (int): Number of sigma values for SO3 diffusion.
            num_omega (int): Number of omega values for SO3 diffusion.
            cache_path (str | Path): Path to the cache directory for SO3 diffusion.
            use_cached_score (bool): Flag to use cached scores for SO3 diffusion.
            L (int): Axis-angle expansion length.
            rot_score_method (Literal["space", "score"]): Method for rotation score calculation.
            rot_score_scaling (Literal["rms", "true"]): How to scale the rotation score.
            reverse_rotations (bool): Deprecated flag for reverse rotations.
            verbose (bool): Flag to enable verbose output.
        kwargs: Additional keyword arguments for future extensions.
        """
        super().__init__()
        # TODO better sampling. Uniform makes noises (sigmas) that are too redundant (always 1 almost...)
        # Note: any R_0_i gives the exact same score-matching loss in Rt!
        self.model = model
        self.include_interactions = include_interactions
        # Denoiser
        self.diffuser = SE3Diffuser(
            min_beta=min_beta,
            max_beta=max_beta,
            schedule=schedule,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            num_omega=num_omega,
            cache_path=Path(cache_path),
            use_cached_score=use_cached_score,
            L=L,
        )

        # Cutoffs (these are only for graph-constriction, no radial basis here only on model forward...)
        self.cutoff_complex_interactions = cutoff_complex_interactions
        self.cutoff_fragment_interactions = cutoff_fragment_interactions
        self.cutoff_complex_virtual = cutoff_complex_virtual

        # Reverse Rotations
        self.reverse_rotations: bool = reverse_rotations
        self.rot_score_method: str = rot_score_method
        if rot_score_method not in ["space", "score"]:
            raise ValueError(f"Invalid rot_score_method: {rot_score_method}. Must be 'space' or 'score'.")
        if rot_score_scaling not in ["rms", "true", None]:
            raise ValueError(f"Invalid rot_score_scaling: {rot_score_scaling}. Must be 'rms', 'true', or None.")
        if rot_score_method == "score" and rot_score_scaling is None:
            raise ValueError(
                "rot_score_scaling must be specified when rot_score_method is 'score'. "
                "Use 'rms' or 'true' to scale the rotation score."
            )
        self.rot_score_scaling: str = rot_score_scaling

        # Print Kwargs if unused with WARNING
        # if len(kwargs) > 0:
        #     print(f"[WARN] Unused kwargs in SigmaDockDenoiser: {kwargs}")

        # Misc
        self.verbose = verbose

    @staticmethod
    def get_flat_fragment_index(batch: Batch) -> torch.Tensor:
        """
        Computes flat [0, sum(F_i))] fragment index for each atom in the batch,
        where F_i is the number of fragments in molecule i.

        Returns:
            flat_frag_idx: LongTensor of shape [N], where N = # atoms.
                        Values are -1 for atoms not assigned to any fragment.
        """
        frag_idx = batch.frag_idx_map  # [N]
        atom_batch_idx = batch.batch  # [N]
        N = frag_idx.size(0)
        B = batch.num_graphs

        # Mask invalids
        valid = frag_idx >= 0
        frag_idx_valid = frag_idx[valid]
        batch_idx_valid = atom_batch_idx[valid]

        # Compute number of fragments per graph
        num_frags_per_graph = torch.zeros(B, dtype=torch.long, device=frag_idx.device)
        for i in range(B):
            idxs = batch_idx_valid == i
            if idxs.any():
                num_frags_per_graph[i] = frag_idx_valid[idxs].max() + 1

        # Offset fragments per graph
        frag_offset = torch.cat(
            [
                torch.tensor([0], device=frag_idx.device),
                num_frags_per_graph.cumsum(dim=0)[:-1],
            ]
        )  # [B]

        # Create flat index
        flat_frag_idx = torch.full((N,), -1, dtype=torch.long, device=frag_idx.device)
        flat_frag_idx[valid] = frag_idx_valid + frag_offset[batch_idx_valid]

        return flat_frag_idx  # [N]

    @staticmethod
    @torch.no_grad()
    def get_fragment_com(pos: torch.Tensor, batch: Data | Batch) -> list[torch.Tensor]:
        """
        Compute per-sample fragment centers of mass (COM), excluding virtual nodes
        and masked dummies. Returns a list of length B (batch size), where each
        entry is a tensor of shape [F_b, 3] containing the COMs for that sample's
        F_b fragments.
        """
        frag_map = batch.frag_idx_map
        node_entity = batch.node_entity
        mask = batch.mask

        # Determine batch splits
        ptr = batch.ptr if isinstance(batch, Batch) else torch.tensor([0, pos.size(0)], device=pos.device)

        com_list = []
        for b in range(ptr.size(0) - 1):
            s, e = int(ptr[b]), int(ptr[b + 1])
            pos_b = pos[s:e]  # [N_b, 3]
            frag_map_b = frag_map[s:e]  # [N_b]
            mask_b = mask[s:e]  # [N_b]
            virt_b = node_entity[s:e] == HPARAMS.get_node_idx("ligand_virtual")  # is_ligand_virtual flag

            # Select real ligand atoms (frag_counter >= 0), excluding virtual and masked dummies
            is_ligand = frag_map_b >= 0
            is_valid = mask_b & (~virt_b)
            if not is_ligand.any():
                com_list.append(torch.empty((0, 3), device=pos.device, dtype=pos.dtype))
                continue

            # Gather positions and corresponding fragment IDs
            pos_sel = pos_b[is_ligand]  # [N_ligand, 3]
            frag_idx_sel = frag_map_b[is_ligand]  # [N_ligand]

            # Unique fragment IDs
            unique_frags = torch.unique(frag_idx_sel)
            # Compute COM per fragment
            coms = []
            for idx in unique_frags:
                # Select atoms belonging to the current fragment
                atom_idx_sel = (frag_idx_sel == idx) & is_valid[is_ligand]
                print(idx, pos_sel[atom_idx_sel].shape)
                # Compute COM for the selected atoms
                coms.append(pos_sel[atom_idx_sel].mean(dim=0))
            com_list.append(torch.stack(coms, dim=0))  # [F_b, 3])

        return com_list

    @staticmethod
    @torch.no_grad()
    def get_fragment_com_and_rot(
        pos: torch.Tensor, batch: Data | Batch
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute per-sample fragment centers of mass (COM) and rotation matrices.
        Excludes virtual nodes and masked dummies. Returns two lists of length B (batch size),
        where each entry is a tensor of shape [F_b, 3] for COMs and [F_b, 3, 3] for rotation matrices.

        Args:
            pos (torch.Tensor): Atom positions of shape [N, 3].
            batch (Data | Batch): Batch object containing graph data.
        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple containing two lists:
                - com_list: List of tensors of shape [F_b, 3] for COMs.
                - rot_list: List of tensors of shape [F_b, 3, 3] for rotation matrices.
        """
        frag_map = batch.frag_idx_map
        node_ent = batch.node_entity
        mask = batch.mask
        I3 = torch.eye(3, device=pos.device, dtype=pos.dtype)
        ptr = batch.ptr if isinstance(batch, Batch) else torch.tensor([0, pos.size(0)], device=pos.device)

        com_list, rot_list = [], []
        for i in range(len(ptr) - 1):
            s, e = int(ptr[i]), int(ptr[i + 1])
            p = pos[s:e]
            fm = frag_map[s:e]
            ne = node_ent[s:e]
            m = mask[s:e]

            valid = (fm >= 0) & (ne != HPARAMS.get_node_idx("ligand_virtual")) & (m)  # drop protein, virtuals, masked
            if not valid.any():
                com_list.append(torch.empty((0, 3), device=pos.device))
                rot_list.append(torch.empty((0, 3, 3), device=pos.device))
                continue

            p_sel = p[valid]
            f_sel = fm[valid]
            fragments = torch.unique(f_sel)

            coms, rots = [], []
            for f in fragments:
                pts = p_sel[f_sel == f]
                com = pts.mean(0)
                X = pts - com
                coms.append(com)

                # Note defining rotation matrix S as identity as it is (and should be) independent in the score.
                if False:
                    if X.size(0) < 3:
                        print(f"[WARN] Fragment {f} has less than 3 atoms. Cannot compute rotation matrix.")
                        rots.append(I3)
                    else:
                        # Compute the covariance matrix
                        C = X.T @ X

                        # Eigenvalue decomposition
                        E, V = torch.linalg.eigh(C)

                        # Sort eigenvalues and corresponding eigenvectors in descending order
                        _, idx = torch.sort(E, descending=True)
                        V = V[:, idx]

                        # Ensure the eigenvectors have the correct sign
                        for k in range(3):
                            col = V[:, k]
                            if col[torch.argmax(col.abs())] < 0:
                                V[:, k] = -col

                        # SVD cleanup to ensure orthogonality and determinant = +1
                        U, S, Vt = torch.linalg.svd(V, full_matrices=False)
                        V = U @ Vt  # Now guaranteed orthonormal with det=+1

                        # Correct for negative determinant if necessary
                        if torch.det(V) < 0:
                            V[:, -1] *= -1

                        rots.append(V)
                else:
                    # Use the identity matrix as a placeholder for rotation
                    rots.append(I3)

            # Stack the lists of COMs and rotations
            com_list.append(torch.stack(coms))
            rot_list.append(torch.stack(rots))

        return com_list, rot_list

    @staticmethod
    @torch.no_grad()
    def get_fragment_mass_inertia(
        pos: torch.Tensor,  # [N,3]
        batch: Data | Batch,
        coms: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each fragment (in the same order as com_list), compute:
        - mass = number of atoms in that fragment
        - inertia tensor about its COM (I = Σ m (|r|² I - r⊗r), here m uniform)
        """
        frag_map = batch.frag_idx_map
        node_ent = batch.node_entity
        mask = batch.mask
        I3 = torch.eye(3, device=pos.device, dtype=pos.dtype)

        # batch→sample splits
        if isinstance(batch, Batch):  # noqa
            ptr = batch.ptr
        else:
            ptr = torch.tensor([0, pos.size(0)], device=pos.device)

        masses = []
        inertias = []

        # walk samples in lockstep with com_list
        batch_size = torch.max(batch.batch).item() + 1 if hasattr(batch, "batch") else 1
        for sample_idx, _ in enumerate(range(batch_size)):
            start, end = int(ptr[sample_idx]), int(ptr[sample_idx + 1])
            p_b = pos[start:end]
            fm_b = frag_map[start:end]
            ne_b = node_ent[start:end]
            m_b = mask[start:end]

            # exactly the same “valid” atoms you use for COM/ROT:
            valid = (fm_b >= 0) & (ne_b != HPARAMS.get_node_idx("ligand_virtual")) & m_b
            if not valid.any():
                continue

            p_sel = p_b[valid]  # [M,3]
            fm_sel = fm_b[valid]  # [M]

            fragments = torch.unique(fm_sel)  # sorted

            # Optionally (for robustness), use provided COMs
            if coms is not None:
                assert len(coms) == batch_size, "COM list length does not match batch size"
                fragment_coms = coms[sample_idx]  # [F_b,3]
            else:
                fragment_coms = None

            # for each fragment f in this sample, in the same order as com_list
            for idx, f in enumerate(fragments):
                pts = p_sel[fm_sel == f]  # [n_f,3]
                # check your COM
                recomputed_com = pts.mean(dim=0)
                if fragment_coms is not None:
                    com = fragment_coms[idx]
                    if not torch.allclose(recomputed_com, com, atol=1e-5):
                        raise RuntimeError(
                            f"COM mismatch sample {sample_idx} frag {f}: {recomputed_com.tolist()} vs {com.tolist()}"
                        )
                else:
                    com = recomputed_com

                m_i = torch.ones(pts.size(0), device=pos.device, dtype=pos.dtype)
                # mass = number of atoms
                masses.append(torch.tensor(float(torch.sum(m_i)), device=pos.device))

                # inertia about COM with uniform mass m_i
                rel = pts - com.unsqueeze(0)  # [n_f,3]
                rr = (rel**2).sum(dim=1)  # [n_f]
                term1 = rr[:, None, None] * I3  # [n_f,3,3]
                term2 = rel[:, :, None] * rel[:, None, :]  # [n_f,3,3]
                I_frag = (term1 - term2).sum(dim=0)  # [3,3]

                # --- robust eigen-clamp regularisation (no in-place) ---
                # user-tunable tiny constants (relative & absolute)
                rel_eps = 1e-8  # relative to inertia scale
                abs_eps = 1e-12  # absolute floor to avoid exactly zero

                # compute scale-aware minimum eigenvalue
                trace = I_frag.trace()  # scalar tensor on correct device/dtype
                min_eig = (rel_eps * (trace / 3.0)).clamp_min(abs_eps)  # scalar tensor

                # eigen-decomposition (3x3; cheap)
                E, V = torch.linalg.eigh(I_frag)  # E: [3], V: [3,3], E ascending

                # clamp small eigenvalues and reconstruct inertia
                E_clamped = E.clamp_min(min_eig)  # [3]
                I_frag_reg = (V * E_clamped.unsqueeze(-2)) @ V.transpose(-1, -2)  # V @ diag(E_clamped) @ V.T

                # append the regularised inertia (no in-place modification)
                inertias.append(I_frag_reg)

        if not masses:
            # no fragments at all
            return torch.empty(0, device=pos.device), torch.empty((0, 3, 3), device=pos.device)

        return torch.stack(masses, dim=0), torch.stack(inertias, dim=0)

    @staticmethod
    @torch.no_grad()
    def get_transformations_from_rototranslations(
        pos0: torch.Tensor,  # [N,3]
        batch: Data,
        T0: torch.Tensor,
        delta_R: torch.Tensor,
        delta_T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-fragment delta rotations & translations to pos0, handling multiple samples.

        Args:
            batch:  Data or Batch, with attributes:
                - pos: [N,3] original atom positions.
                - frag_idx_map: [N] long, local fragment ID per sample (>=0 for ligand, -1 for protein).
                - batch: [N] long, sample index per atom (only on Batch).
            T0:     [F_total,3]  concatenated COMs for all fragments in batch order.
            delta_R:[F_total,3,3] per-fragment rotation deltas.
            delta_T:[F_total,3]  per-fragment translation deltas.

        Returns:
            pos_new: [N,3] updated positions after applying the deltas.

        New pos[i] = R[f_global] @ (pos_old[i] - T0[f_global]) + T0[f_global] + delta_T[f_global]
        for each atom i with f_local = frag_idx_map[i] >= 0, and
        f_global = base_offset[sample_i] + f_local.
        Protein atoms (f_local < 0) are left unchanged.
        """
        frag_local = batch.frag_idx_map
        device = pos0.device

        # sample index per atom (zeros if no batch attr)
        sample_idx = batch.batch if hasattr(batch, "batch") else torch.zeros_like(frag_local, dtype=torch.long)

        # mask ligand atoms
        is_lig = frag_local >= 0
        if not is_lig.any():
            return pos0

        f_local = frag_local[is_lig]
        s_idx = sample_idx[is_lig]

        # count fragments per sample
        B = int(sample_idx.max().item()) + 1
        frag_counts = []
        for b in range(B):
            sel = (sample_idx == b) & (is_lig)
            if sel.any():
                frag_counts.append(int(frag_local[sel].max().item()) + 1)
            else:
                frag_counts.append(0)
        # prefix sum offsets
        offsets = torch.tensor([0, *frag_counts[:-1]], device=device).cumsum(dim=0)

        # global fragment index for each atom
        f_global = offsets[s_idx] + f_local  # [M]

        # gather deltas and COM
        R_g = delta_R[f_global]  # [M,3,3]
        T0_g = T0[f_global]  # [M,3]
        dT_g = delta_T[f_global]  # [M,3]

        pts = pos0[is_lig]  # [M,3]
        rel = pts - T0_g  # [M,3]

        # apply rotation and translation
        rel_rot = torch.matmul(R_g, rel.unsqueeze(-1)).squeeze(-1)  # [M,3]
        new_pts = rel_rot + T0_g + dT_g  # [M,3]

        pos_new = pos0.clone()
        pos_new[is_lig] = new_pts
        return pos_new

    @staticmethod
    @torch.no_grad()
    def get_local_graph(
        pos: torch.Tensor,
        batch: Data | Batch,
        cutoff_complex_interactions: float,
        cutoff_complex_virtual: float,
        cutoff_fragments: float,
        lig_just_atoms: bool = True,
        edge_dim: int = 4,
    ) -> Data | Batch:
        """
        Reset local edges & ligand positions in the batch.
        Args:
            batch (Data | Batch): The input data batch.
            pos (torch.Tensor): The new positions to set in the batch.
            cutoff_complex_interactions (float | None): The cutoff distance (A) for complex interactions.
            cutoff_complex_virtual (float | None): The cutoff distance (A) for complex virtual interactions.
            cutoff_fragments (float | None): The cutoff distance (A) for fragment interactions.
            lig_just_atoms (bool): Flag to indicate if only ligand atoms should be considered for complex.
        Returns:
            Data | Batch: The updated batch with reset local edges.
        """
        assert "node_entity" in batch, "Batch must contain node_entity attribute."
        assert "frag_idx_map" in batch, "Batch must contain frag_idx_map attribute."

        # Protein Ligand Local Interaction Edges
        dynamic_interaction_edges = get_local_interactions_batched(
            pos=pos,
            batch=batch.batch,
            node_entity=batch.node_entity,
            frag_idx_map=batch.frag_idx_map,
            cutoff_complex_interactions=cutoff_complex_interactions,
            cutoff_complex_virtual=cutoff_complex_virtual,
            cutoff_fragments=cutoff_fragments,
            lig_just_atoms=lig_just_atoms,
            edge_dim=edge_dim,
        )
        return dynamic_interaction_edges

    @staticmethod
    @torch.no_grad()
    def merge_and_process_edges(
        global_graph: dict[str, torch.Tensor],
        local_graph: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Merge a local interaction graph, optionally pruning edges touching masked nodes,
        and update each graph's node positions in-place.

        Batch object is NOT updated in-place (safer), but a new dict is returned.

        Args:
            batch (Batch): A PyG Batch containing existing edge_index, edge_attr, edge_entity, and optional mask.
            local_graph (dict): Dictionary with keys 'edge_index', 'edge_attr', 'edge_entity' for new edges.
            prune_full (bool): If True, remove any edge where either endpoint node is masked (batch.mask == True).

        Returns:
            dict[str, Tensor]: Updated edges.
        """
        # Unpack new edges
        ei_new = local_graph["edge_index"]
        ea_new = local_graph["edge_attr"]
        ee_new = local_graph["edge_entity"]

        # Concatenate existing and new edges
        ei = torch.cat([global_graph["edge_index"], ei_new], dim=1)
        ea = torch.cat([global_graph["edge_attr"], ea_new], dim=0)
        ee = torch.cat([global_graph["edge_entity"], ee_new], dim=0)

        # Prune edges touching masked nodes if requested
        if mask is not None:
            raise DeprecationWarning("Masking here is deprecated and will be removed in future versions.")
            src, dst = ei
            keep = mask[src] & mask[dst]
            ei = ei[:, keep]
            ea = ea[keep]
            ee = ee[keep]

        return {"edge_index": ei, "edge_attr": ea, "edge_entity": ee}

    def sample_time(self, num_samples: int) -> torch.Tensor:
        """
        Sample a time step for the diffusion process.
        Args:
            num_samples (int): The number of samples to generate.
        Returns:
            torch.Tensor: The sampled time tensor.
        """
        # Uniform distribution from [epsilom_t, 1.0]
        # Note since we bucketize on "right" we will never sample t=0.0 but rather on 1/N (N~2000)
        uniform = torch.rand(num_samples)  # [B]
        # NOTE FUTURE potentially we could pass in [sigma_t, sigma_r] to the model instead of the time.
        return HPARAMS.general.epsilon_t + (1 - HPARAMS.general.epsilon_t) * uniform  # [B]

    def _prepare_batch(self, batch: Data | Batch) -> Batch:
        """
        Move batch data to the model's device.

        Args:
            batch (Data | Batch): Input batch.

        Returns:
            Batch: Batch on correct device.
        """
        device = next(self.model.parameters()).device
        if batch.x.device != device:
            print("[WARN] Moving batch to model device.")
            batch = batch.to(device)
        return batch

    def _sample_time_and_sigma(self, batch: Batch | Data) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute per-graph element batch sizes, sample diffusion times, and obtain sigmas.

        Args:
            batch (Batch): Input batch with ptr or single Data.

        Returns:
            batch_ptrs (torch.Tensor): Number of points per graph [B].
            t (torch.Tensor): Sampled times [B].
            sigmas (dict[str, torch.Tensor]): Sigma values for translation and rotation [B, ...].
        """
        if isinstance(batch, Batch):
            ptr = batch.ptr
            batch_ptrs = ptr[1:] - ptr[:-1]  # [B]
        else:
            batch_ptrs = batch.x.shape[0] * torch.ones(1, device=batch.x.device, dtype=torch.long)  # [N]

        # Sample time steps for each graph (independent) in the batch
        t = self.sample_time(len(batch_ptrs)).to(batch.x)  # [B]
        sigmas: dict[str, torch.Tensor] = self.diffuser.sigma(t)  # [B]
        return t, sigmas

    def _get_initial_states(self, batch: Batch | Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Center and normalize coordinates, then compute fragment centers of mass and rotations.

        Args:
            batch (Batch): Input batch with ref_pos and pocket_com.

        Returns:
            pos_0 (torch.Tensor): Normalized initial positions [N,3].
            T0 (torch.Tensor): Concatenated fragment centers [B*F,3].
            R0 (torch.Tensor): Concatenated rotation matrices [B*F,3,3].
        """
        if isinstance(batch, Batch):
            batch_pointer = batch.ptr[1:] - batch.ptr[:-1]  # [B]
        else:
            batch_pointer = batch.x.shape[0] * torch.ones(1, device=batch.x.device, dtype=torch.long)  # [N]
        # Remove Pocket COM from reference positions
        pos_0 = batch.ref_pos - batch.pocket_com.repeat_interleave(batch_pointer, dim=0)
        # Normalize positions
        pos_0 = pos_0 / HPARAMS.general.dimensional_scale
        # Reference rotations and translations
        T0_list, R0_list = self.get_fragment_com_and_rot(pos_0, batch)
        num_fragments = torch.tensor([len(t) for t in T0_list], device=batch.x.device)  # [B]
        return (
            pos_0,
            torch.cat(T0_list, dim=0),
            torch.cat(R0_list, dim=0),
            num_fragments,
        )

    def _sample_diffusion(
        self,
        T0: torch.Tensor,
        R0: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Broadcast time and sigmas across fragments, then sample forward marginal diffusion.

        Args:
            T0 (torch.Tensor): Initial fragment centers [B*F,3].
            R0 (torch.Tensor): Initial rotations [B*F,3,3].
            t_batch (torch.Tensor): Times per fragment [BxF].

        Returns:
            sampled (dict): Contains 'R_t', 'T_t', 'T_score', 'R_score'.
        """
        sampled = self.diffuser.forward_marginal(T0, R0, t_batch)
        return sampled

    def _apply_transformations(
        self,
        pos_0: torch.Tensor,
        batch: Batch,
        T0: torch.Tensor,
        R0: torch.Tensor,
        R_t: torch.Tensor,
        T_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute transformed positions from roto-translations.

        Args:
            pos_0 (torch.Tensor): Original positions [N,3].
            batch (Batch): Original batch.
            T0 (torch.Tensor): Initial fragment centers [B*F,3].
            R0 (torch.Tensor): Initial rotations [B*F,3,3].
            sampled_diffusion (dict): Output of forward_marginal ('R_t','T_t').

        Returns:
            pos_t (torch.Tensor): Transformed positions [N,3].
        """
        # Compute deltas against initial states
        delta_R = R_t @ R0.transpose(-1, -2)  # [B*F,3,3]
        delta_T = T_t - T0  # [B*F,3]
        pos_t = self.get_transformations_from_rototranslations(pos_0, batch, T0, delta_R, delta_T)  # [N,3]
        return pos_t

    def _update_batch(
        self,
        batch: Batch,
        pos_0: torch.Tensor,
        pos_t: torch.Tensor,
        prune: bool = False,
    ) -> Batch:
        """
        Update batch with pos_0 and pos_t, recompute edges if needed.

        Args:
            batch (Batch): Input batch with original global edges and mask.
            pos_0 (torch.Tensor): Original positions [N,3].
            pos_t (torch.Tensor): Transformed positions [N,3].
        Returns:
            Batch: Updated batch with new positions and edges.
        """
        batch["pos_0"] = pos_0
        batch["pos_t"] = pos_t

        # Remove existing interaction edges if they exist (removes all "local_dynamic" edges)
        self._prune_local_edges(batch)
        assert set(HPARAMS.get_edge_group_indices("local_dynamic")) not in set(
            batch.edge_entity.unique().cpu().numpy()
        ), "Failed to prune local dynamic edges from batch."
        # Extract global edges
        global_edges = {
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "edge_entity": batch.edge_entity,
        }

        # Add new interaction edges based on updated positions
        interaction_edges = self._compute_interaction_edges(batch, pos_t)
        # Merge and filter (Optionally remove edges touching masked nodes ->
        complex_edges: dict[str : torch.Tensor] = self.merge_and_process_edges(
            global_edges,
            interaction_edges,
        )
        replace_batch_edge_attribute(batch, complex_edges)

        # Prune edges touching masked nodes. Defaults to False so that this happens at the model forward instead.
        if prune and hasattr(batch, "mask") and batch.mask is not None:
            src, dst = batch.edge_index
            keep = batch.mask[src] & batch.mask[dst]
            batch.edge_index = batch.edge_index[:, keep]
            batch.edge_attr = batch.edge_attr[keep]
            batch.edge_entity = batch.edge_entity[keep]
        return batch

    def _compute_interaction_edges(
        self,
        batch: Batch,
        pos_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate and merge local and global graph edges based on updated positions.

        Args:
            batch (Batch): Batch with original global edges and mask.
            pos_t (torch.Tensor): Transformed node positions [N,3].

        Returns:
            dict[str, torch.Tensor]: Merged edge_index, edge_attr, and edge_entity tensors.
        """
        # Build local fragment interactions
        local_dynamic_edges = self.get_local_graph(
            pos_t,
            batch,
            cutoff_complex_interactions=self.cutoff_complex_interactions / HPARAMS.general.dimensional_scale,
            cutoff_complex_virtual=self.cutoff_complex_virtual / HPARAMS.general.dimensional_scale,
            cutoff_fragments=self.cutoff_fragment_interactions / HPARAMS.general.dimensional_scale,
            edge_dim=batch.edge_attr.shape[1] if hasattr(batch, "edge_attr") else 4,
        )

        return local_dynamic_edges

    def _prune_local_edges(self, batch: Batch) -> Batch:
        """
        Remove existing local dynamic edges from the batch.

        Args:
            batch (Batch): Input batch with existing edges.

        Returns:
            Batch: Batch with interaction and other local edges removed.
        """
        assert hasattr(batch, "edge_index") and hasattr(batch, "edge_entity"), (
            "Batch must contain edge_index and edge_entity attributes."
        )
        # Filter out edges that are not complex or fragments
        # mask = (batch.edge_entity != EDGE_ENTITY["inter_complex"]) & (
        #     batch.edge_entity != EDGE_ENTITY["inter_fragments"]
        # )
        cutoff_edge_indices = HPARAMS.get_edge_group_indices("local_dynamic")
        mask = ~torch.isin(
            batch.edge_entity,
            torch.tensor(cutoff_edge_indices, device=batch.edge_entity.device),
        )
        # Prune out edges at current time with (old) dynamic interactions which change through time
        batch.edge_index = batch.edge_index[:, mask]
        batch.edge_attr = batch.edge_attr[mask]
        batch.edge_entity = batch.edge_entity[mask]
        return batch

    def _compute_forces(
        self,
        batch: Batch,
        t: torch.Tensor,
        **model_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run model to get ligand pseudo-forces at non-masked nodes.

        Args:
            batch (Batch): Input batch with node_entity and mask.
            t (torch.Tensor): Sampled time tensor [B].
            **model_kwargs: Additional keyword arguments for the model.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - lig_forces: Pseudo forces at non-masked nodes [NL, 3].
                - forces_idxs: Indices of the non-masked nodes [NL].
        """
        # NOTE pseudo forces ~ \epsilon \in N(0,1). Scaled by sigma for trans and rotation to get pseudo forces and pseudo torques. # noqa
        epsilon_theta = self.model(batch, t, **model_kwargs)  # [B x F x A, 3, 3]
        # Ensure pseudo_forces are selected at the NON-MASKED NODES
        forces_idxs = get_lig_idxs(batch.node_entity, mask=batch.mask)  # [B x NL: <F x A>]
        # NOTE negating the forces here is equivalent to negating the epsilon_theta in the model.
        lig_forces = -epsilon_theta.index_select(0, forces_idxs)  # [B x NL, 3]
        assert batch.mask[forces_idxs].all() & (lig_forces.shape[0] == forces_idxs.shape[0]), (
            "Pseudo forces are not selected at non-masked nodes."
        )
        return lig_forces, forces_idxs

    @staticmethod
    def linear_mechanics(
        pos_atoms: torch.Tensor,  # [K,3]
        forces: torch.Tensor,  # [K,3]
        frag_idx_continuous: torch.Tensor,  # [K]
        coms: torch.Tensor,  # [M,3] centers of mass at time t
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bin per atom forces into per fragment total force & torque about updated COMs.

        Args:
            pos_atoms            Tensor[K,3] atom positions at time t
            forces               Tensor[K,3] per atom force vectors at time t
            frag_idx_continuous  Tensor[K]   flat fragment index per atom
            coms                 Tensor[M,3] fragment centers of mass at time t

        Returns:
            total_force  Tensor[M,3]
            total_torque Tensor[M,3]
        """
        num_fragments = coms.size(0)  # [M]
        device, dtype = forces.device, forces.dtype

        # initialize accumulators
        total_force = torch.zeros((num_fragments, 3), device=device, dtype=dtype)
        total_torque = torch.zeros((num_fragments, 3), device=device, dtype=dtype)

        # sum up forces -> Same op as scatter(total_force, frag_idx_continuous, reduce = "sum")
        total_force = total_force.index_add(0, frag_idx_continuous, forces)

        # compute relative positions to COM and torque contributions
        rel = pos_atoms - coms[frag_idx_continuous]  # [K,3]
        torque_terms = torch.cross(rel, forces, dim=1)  # [K,3]
        total_torque = total_torque.index_add(0, frag_idx_continuous, torque_terms)

        return total_force, total_torque

    @staticmethod
    def newton_maruyama(
        force_per_fragment: torch.Tensor,  # [M,3]
        torque_per_fragment: torch.Tensor,  # [M,3]
        mass: torch.Tensor,  # [M]
        inertia: torch.Tensor,  # [M,3,3]
        reverse_rotations: bool = False,  # whether to reverse rotations
    ) -> dict[str : torch.Tensor]:
        """
        Convert per-fragment force & torque into rigid-body translation and rotation updates,
        using so(3) hat and matrix exponential.

        Args:
            force_per_fragment (torch.Tensor): Per-fragment forces [M, 3].
            torque_per_fragment (torch.Tensor): Per-fragment torques [M, 3].
            mass (torch.Tensor): Masses of fragments [M].
            inertia (torch.Tensor): Inertia tensors of fragments [M, 3, 3].
            R_t (torch.Tensor): Rotation matrices at time t [M, 3, 3].
            r3_scaling (torch.Tensor | None): Scaling factor for translation updates [M].
            so3_scaling (torch.Tensor | None): Scaling factor for rotation updates [M].
        Returns:
            dict[str : torch.Tensor]: Dictionary containing:
                - total_force: [M, 3] translation increments.
                - delta_W: [M, 3] angular increments.
                - omega: [M, 3, 3] skew-symmetric matrices.
        """
        # NOTE pre-scaled force & torques.

        # 1) translation increment: ΔT = F / m
        dT = force_per_fragment / mass[..., None]  # [M,3]

        # 2) angular increment dW = I^{-1} T
        # Regularize inertia tensor for numerical stability (done already in get_fragment_mass_inertia)
        inertia = inertia + torch.eye(3, device=inertia.device, dtype=inertia.dtype) * 1e-8

        # NOTE this is the LOCAL (right-invariant - BODY) frame of reference, so we can use the inertia tensor directly.
        # NOTE if calculating torque in local frame, effectively tansppsing torque isomorphism which is equivalent
        # dW_global = torch.einsum('...ij,...j->...i', R_t, dW_local)
        # Transport into GLOBAL (left-invariant - WORLD) frame of reference like the rest of the code.
        dW_world = torch.linalg.solve(inertia, torque_per_fragment.unsqueeze(-1)).squeeze(-1)  # [M,3]

        # 3) lift to so(3): skew-symmetric generator
        # NOTE We need to keep the left-invariant convention of dR Represents a global rotation
        if reverse_rotations:
            print("[WARN] Reverse rotations is deprecated and will be removed in future versions.")
            omega = so3_utils.hat(-dW_world)  # [...,3,3]
        else:
            omega = so3_utils.hat(dW_world)  # [...,3,3]

        # Clamp omega to reasonable values
        omega = torch.clamp(omega, min=-1e3, max=1e3)

        return {
            "total_force": dT,
            "delta_W": dW_world,
            "omega": omega,
        }

    def _compute_fragment_dynamics(
        self,
        batch: Batch,
        R_t: torch.Tensor,  # [BF,3,3]
        R0: torch.Tensor,  # [BF,3,3]
        T_t: torch.Tensor,  # [BF,3]
        lig_forces: torch.Tensor,  # [NL,3]
        forces_idxs: torch.Tensor,  # [NL]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-fragment total forces and torques via linear mechanics.

        Args:
            batch (Batch): Updated batch containing pos_t.
            sampled_diffusion (dict): Diffusion outputs including 'R_t' and 'T_t'.
            lig_forces (torch.Tensor): Ligand pseudo-forces [NL,3].
            forces_idxs (torch.Tensor): Indices of ligand nodes [NL].
            R0 (torch.Tensor): Initial rotations [BF,3,3].

        Returns:
            force_per_fragment (torch.Tensor): [BF,3]
            torque_per_fragment (torch.Tensor): [BF,3]
            frag_mass (torch.Tensor): [B,F]
            frag_inertia_t (torch.Tensor): [BF,3,3]
        """
        # Extract transforms
        pos_0, pos_t = batch["pos_0"], batch["pos_t"]

        # Mass & inertia at t0, then at time t
        M, I_0 = self.get_fragment_mass_inertia(pos_0, batch)
        _, I_t = self.get_fragment_mass_inertia(pos_t, batch)
        # Ensure there are no fragments with mass below 2 (makes force-inertia incompatible)
        assert M.min() >= 2, "Fragment mass below 2 detected. Check input data."

        # Compute delta R_t for inertia transformation
        if self.verbose:
            delta_R_t = R_t @ R0.transpose(-1, -2)
            I_t_transformed = delta_R_t @ I_0 @ delta_R_t.transpose(-1, -2)
            if not torch.allclose(I_t_transformed, I_t, rtol=1e-2, atol=1e-1):
                print(
                    "[WARNING] Inertia tensors at do not match transformed t->0 inertia tensors. Using time t inertia."
                )
                print(
                    f"Max diff: {torch.max(torch.abs(I_t_transformed - I_t)):.4f}, \
                    Mean diff: {torch.mean(torch.abs(I_t_transformed - I_t)):.4f}"
                )

        # Compute actual forces & torques per fragment
        cont_idx = self.get_flat_fragment_index(batch)
        force_per_fragment, torque_per_fragment = self.linear_mechanics(
            pos_atoms=pos_t[forces_idxs],  # [B x NL, 3]
            forces=lig_forces,  # [B x NL, 3]
            frag_idx_continuous=cont_idx[forces_idxs],  # [B x NL]
            coms=T_t,  # [B x F, 3] fragment centers at time t
        )  # [B x F, 3], [B x F, 3]
        return force_per_fragment, torque_per_fragment, M, I_t

    def _get_scalings(self, t_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get scaling factors for forces and torques based on fragment masses and inertia.

        Args:
            t_batch (torch.Tensor): Times per fragment [BF].

        Returns:
            tuple: (r3_scaling, so3_scaling) where:
                - r3_scaling (torch.Tensor): Scaling factor for translational updates.
                - so3_scaling (torch.Tensor): Scaling factor for rotational updates.
        """
        # Scale according to diffusion schedule
        r3_scaling = self.diffuser._r3_diffuser.score_scaling(t_batch).clamp(min=1e-3, max=1e3)
        so3_scaling = self.diffuser._so3_diffuser.score_scaling(t_batch).clamp(min=1e-3, max=1e3)
        return r3_scaling, so3_scaling

    def _predict_fragment_updates(
        self,
        force_per_fragment: torch.Tensor,
        torque_per_fragment: torch.Tensor,
        frag_mass: torch.Tensor,
        frag_inertia_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Apply Newton-Maruyama to get updated force and rotational increments.

        Args:
            force_per_fragment (torch.Tensor): [BF,3]
            torque_per_fragment (torch.Tensor): [BF,3]
            frag_mass (torch.Tensor): [B,F]
            frag_inertia_t (torch.Tensor): [BF,3,3]
            t_batch (torch.Tensor): Times per fragment [BF].

        Returns:
            dict with keys 'total_force', 'delta_R' (and optionally others)
        """
        return self.newton_maruyama(
            force_per_fragment=force_per_fragment,  # [B x F, 3]
            torque_per_fragment=torque_per_fragment,  # [B x F, 3, 3]
            mass=frag_mass,  # [B x F]
            inertia=frag_inertia_t,  # [B x F, 3, 3]
            reverse_rotations=self.reverse_rotations,  # whether to reverse rotations (for depreceated compatibility)
        )  # [B, 3], [B, 3, 3]

    def _compute_scores(
        self,
        sampled: dict[str, torch.Tensor],
        updates: dict[str, torch.Tensor],
        t_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Convert fragment updates into final translation and rotation scores.

        Args:
            sampled (dict[str, torch.Tensor]): Sampled diffusion outputs containing 'T_t' and 'R_t'.
            updates (dict[str, torch.Tensor]): Fragment updates containing 'total_force' and 'delta_R'.
            t_batch (torch.Tensor): Times per fragment [BxF].
        Returns:
            dict[str, torch.Tensor]: A dictionary containing predicted scores and updated fragment states.
        """
        # Get Scalings
        r3_scaling, so3_scaling = self._get_scalings(t_batch)  # [B x F], [B x F]
        T_t, R_t = sampled["T_t"], sampled["R_t"]

        # Compute translational scores & update
        pred_T_score = updates["total_force"] * r3_scaling[..., None]  # [B*F,3]
        T0_hat = self.diffuser.calc_trans_0(pred_T_score, T_t, t_batch)

        # Compute rotational scores & update
        if self.rot_score_method == "space":
            scaled_omega = updates["omega"] * so3_scaling[..., None, None]  # [B*F,3,3]
            dR = so3_utils.exp(scaled_omega)  # [...,3,3]
            R0_hat = dR @ R_t
            pred_R_score = self.diffuser.calc_rot_score(R_t, R0_hat, t_batch)
        elif self.rot_score_method == "score":
            # NOTE: we translate omega which is skew-symmetric at the identity into global world frame at R_t.
            if self.rot_score_scaling == "true":
                # Compute alpha(t, w):
                omega_scalar = torch.linalg.norm(so3_utils.vee(updates["omega"]), 2, dim=-1)
                zero_mask = omega_scalar < 1e-6
                if torch.any(zero_mask):
                    # Create a new omega with tiny random noise for the zero entries
                    # The vee/hat operations move between the 3-vector and 3x3 matrix form
                    omega_vec = so3_utils.vee(updates["omega"])

                    # Generate noise only for the entries that need it
                    noise = torch.randn_like(omega_vec[zero_mask])

                    # Add scaled noise to the zero vectors
                    omega_vec[zero_mask] += noise * 1e-6  # A small constant factor

                    # Re-apply the hat operator and re-calculate the norm
                    updates["omega"] = so3_utils.hat(omega_vec)
                    omega_scalar = torch.linalg.norm(omega_vec, 2, dim=-1)
                sigma = self.diffuser._so3_diffuser.sigma(t_batch)
                d_log_f = d_log_f_d_omega(omega_scalar, sigma, L=self.diffuser._so3_diffuser.L)
                safe_omega_scalar = omega_scalar.clone()
                safe_omega_scalar[safe_omega_scalar < 1e-6] = 1.0  # harmless dummy value to avoid div-by-zero in grad
                alpha = d_log_f / safe_omega_scalar
                alpha = torch.where(omega_scalar < 1e-6, -1.0 / (sigma**2 + 1e-8), alpha)
            elif self.rot_score_scaling == "rms":
                alpha = so3_scaling
            # This is equivalent to the score of the Rt, we muliply scalar (alpha(t, w)) with -omega_hat, R_t
            pred_R_score = -alpha[:, None, None] * updates["omega"] @ R_t
            R0_hat = self.diffuser._so3_diffuser.reverse(sampled["R_t"], pred_R_score, t_batch, t_batch, noise_scale=0)
        else:
            raise ValueError(f"Unknown rot_score_method: {self.rot_score_method}. Choose 'space' or 'score'.")
        return {
            "pred_T_score": pred_T_score,
            "pred_R_score": pred_R_score,
            "T0_hat": T0_hat,
            "R0_hat": R0_hat,
        }

    def _compute_true_scores(
        self,
        T0: torch.Tensor,
        R0: torch.Tensor,
        Tt: torch.Tensor,
        Rt: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute true scores for translation and rotation based on the batch data.

        Args:
            T0 (torch.Tensor): Initial fragment centers [B*F,3].
            R0 (torch.Tensor): Initial rotations [B*F,3,3].
            Tt (torch.Tensor): Transformed fragment centers at time t [B*F,3].
            t_batch (torch.Tensor): Sampled time tensor [B*F].

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the true scores.
        """
        # Compute true translational scores
        true_T_score = self.diffuser.calc_trans_score(Tt, T0, t_batch)
        # Compute true rotational scores
        true_R_score = self.diffuser.calc_rot_score(Rt, R0, t_batch)
        return {
            "true_T_score": true_T_score,
            "true_R_score": true_R_score,
        }

    def forward(
        self,
        batch: Data | Batch,
        **model_kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the denoiser. Used only for training.
        Args:
            batch (Data | Batch): The input data batch.
            model_kwargs (dict[str, Any]): Additional keyword arguments for the model.
        Returns:
            dict[str, torch.Tensor]: A dictionary containing the output tensors.
        """

        # Move batch to the same device as the model
        batch = self._prepare_batch(batch)

        # Sample time and get sigmas
        t, sigmas = self._sample_time_and_sigma(batch)  # [B], [B], {T: [B], R: [B]}

        # Normalize coordinates and compute initial fragment COM & rotations
        pos_0, T0, R0, num_fragments = self._get_initial_states(batch)  # [N,3], [BxF,3], [BxF,3,3]
        # Broadcast time to all fragments
        t_batch = t.repeat_interleave(num_fragments)  # [BxF]

        # Noise the current denoised states and get the scores. Broadcast time to all fragments.
        sampled_diffusion = self._sample_diffusion(T0=T0, R0=R0, t_batch=t_batch)

        # Apply transformations to get new positions at time t
        pos_t = self._apply_transformations(
            pos_0=pos_0,
            batch=batch,
            T0=T0,
            R0=R0,
            R_t=sampled_diffusion["R_t"],
            T_t=sampled_diffusion["T_t"],
        )  # [N,3]

        # Update batch with new positions (also normalized pos_0) and optional interactions
        batch = self._update_batch(batch=batch, pos_0=pos_0, pos_t=pos_t)

        # Compute the forces and indices for the ligands
        lig_pseudoforces, forces_idxs = self._compute_forces(
            batch=batch, t=t, **model_kwargs
        )  # [B x F x A, 3], [B x F x A]

        # Linear mechanics: mass, inertia, force & torque
        force_per_fragment, torque_per_fragment, frag_mass, frag_inertia_t = self._compute_fragment_dynamics(
            batch=batch,
            R_t=sampled_diffusion["R_t"],  # [B x F, 3, 3]
            T_t=sampled_diffusion["T_t"],  # [B x F, 3]
            R0=R0,  # [B x F, 3, 3]
            lig_forces=lig_pseudoforces,
            forces_idxs=forces_idxs,
        )  # [B x F, 3], [B x F, 3], [B,F], [B x F, 3, 3]

        # Scale forces/torques and predict updates (Newton-Maruyama)
        fragment_updates = self._predict_fragment_updates(
            force_per_fragment=force_per_fragment,
            torque_per_fragment=torque_per_fragment,
            frag_mass=frag_mass,
            frag_inertia_t=frag_inertia_t,
        )  # [B x F, 3], [B x F, 3, 3]
        r3_scaling, so3_scaling = self._get_scalings(t_batch)  # [B x F], [B x F]

        # Compute translational and rotational scores
        scores = self._compute_scores(sampled_diffusion, fragment_updates, t_batch)

        return {
            # RotoTranslations
            "t": t,  # [B]
            "t_batch": t_batch,  # [BxF]
            "sigma_T": sigmas["T"],  # [BxF]
            "sigma_R": sigmas["R"],  # [BxF]
            "num_fragments": num_fragments,  # [B]
            "pos_0": pos_0,  # [N,3]
            "pos_t": pos_t,  # [N,3]
            "T_0": T0,  # [B x F, 3]
            "R_0": R0,  # [B x F, 3, 3]
            "T_0_hat": scores["T0_hat"],  # [B x F, 3]
            "R_0_hat": scores["R0_hat"],  # [B x F, 3, 3]
            # Model Preds
            "pseudoforces": lig_pseudoforces,  # [B x F x A, 3]
            "force_per_fragment": force_per_fragment,  # [B x F, 3]
            "torque_per_fragment": torque_per_fragment,  # [B x F, 3]
            # Scores
            "pred_T_score": scores["pred_T_score"],  # [B x F, 3]
            "pred_R_score": scores["pred_R_score"],  # [B x F, 3, 3]
            "true_T_score": sampled_diffusion["T_score"],  # [B x F, 3]
            "true_R_score": sampled_diffusion["R_score"],  # [B x F, 3, 3]
            # Scalings are INDEPENDENT of the DIMENSIONAL SCALE and therefore we need an additional scaling factor!
            "T_score_scaling": r3_scaling,  # [B x F]
            "R_score_scaling": so3_scaling,  # [B x F]
        }

    def compute_losses(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute the four core losses:
          - Translation score MSE
          - Rotation score MSE
          - Fragment COM reconstruction loss
          - Fragment rotation reconstruction loss

        Args:
            out (dict[str, torch.Tensor]): The output dictionary from the forward pass.
        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed losses.
        """
        t_batch = out["t_batch"]  # [B x F]

        #  ============ SCORE LOSS ============ #
        pt, tt = out["pred_T_score"], out["true_T_score"]
        lambda_trans_score = 1 / out["T_score_scaling"] ** 2
        # NOTE adding alpha_t also? Equivalent to lambda_t = sigma ** 2 / alpha
        alpha_trans_t = self.diffuser._r3_diffuser.input_scaling(t_batch)
        batch_trans_score_loss = (
            # lambda_trans_score / alpha_trans_t * ((pt - tt).pow(2).sum(-1))
            lambda_trans_score * ((pt - tt).pow(2).sum(-1))
        )  # [B x F]

        pr, tr = out["pred_R_score"], out["true_R_score"]
        lambda_rot = 1 / out["R_score_scaling"] ** 2
        # Frobenius norm of the Lie algebra element disentangles S (local rot matx from R_0' = R_0 @ S).
        err = pr - tr
        batch_rot_score_loss = lambda_rot * (err * err).sum(dim=(-1, -2))  # [B x F]
        # batch_rot_score_loss = lambda_rot * torch.linalg.norm(err, "fro", dim=(-1, -2)) ** 2  # [B x F]

        #  ============ DATA SPACE LOSSES ============ #
        T0, T0h = out["T_0"], out["T_0_hat"]
        # Equivalent to alpha(t) / sigma_t ** 2
        # lambda_trans_data = alpha_trans_t * out["T_score_scaling"] ** 2
        lambda_trans_data = (alpha_trans_t * out["T_score_scaling"]) ** 2
        batch_T0_loss = lambda_trans_data * ((T0 - T0h).pow(2).sum(-1))

        R0, R0h = out["R_0"], out["R_0_hat"]
        rel = R0h.transpose(-2, -1) @ R0
        log_rel = so3_utils.log(rel)  # [B x F, 3, 3]
        lambda_rot_data = out["R_score_scaling"] ** 2
        # Geodesics in so(3) are given by the norm of the Lie algebra element.
        batch_R0_loss = lambda_rot_data * ((log_rel**2).sum(dim=(-1, -2)))  # [B x F]
        # batch_R0_loss = lambda_rot_data * (log_rel.norm(dim=(-1, -2), p="fro").pow(2))  # [B x F]

        return {
            # Batch per-fragment loss
            "T_score": batch_trans_score_loss,  # [B x F]
            "R_score": batch_rot_score_loss,  # [B x F]
            "T0": batch_T0_loss,  # [B x F]
            "R0": batch_R0_loss,  # [B x F]
        }

    def scaled_fragmented_loss(
        self,
        losses: dict[str, torch.Tensor],
        num_fragments: torch.Tensor,  # [B]
        fragment_scaling: float,
    ) -> dict[str, torch.Tensor]:
        """
        Process the losses to get the final loss accoridng to fragmentation count scatter-sum.
        Args:
            losses (dict[str, torch.Tensor]): The input losses dictionary.
            num_fragments (torch.Tensor): The number of fragments in the batch.
            fragment_scaling (float): The scaling factor for the fragment loss.
        Returns:
            dict[str, torch.Tensor]: The processed losses dictionary.
        """
        batch_idx = torch.arange(num_fragments.shape[0], device=num_fragments.device).repeat_interleave(num_fragments)

        # Ensure loss values have size [B x F]
        assert all(x.shape[0] == batch_idx.shape[0] for k, x in losses.items()), ValueError(
            f"Loss has shapeshould be [B x F] with shape {batch_idx.shape}."
        )

        # Scattered losses
        scaled_losses: dict[str : torch.Tensor] = {
            k: scatter(x, batch_idx, reduce="sum") / (num_fragments**fragment_scaling) for k, x in losses.items()
        }
        return scaled_losses

    def __repr__(self) -> str:
        """
        String representation of the SigmaDockDenoiser class.
        Returns:
            str: The string representation of the class.
        """
        return f"SigmaDockDenoiser({self.model.__class__.__name__})"
