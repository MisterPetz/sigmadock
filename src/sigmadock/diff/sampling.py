from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import seed_everything

# from torch import nn
from torch_geometric.data import Batch
from tqdm import tqdm

from sigmadock.diff.se3_diffuser import SE3Diffuser
from sigmadock.oracle import HPARAMS


# NOTE this function evaluates given true pose exists (ReDocking Scenario) -> Not generic (yet)
def sample_notebook(
    denoiser: SE3Diffuser,
    batch: Batch,
    t_min: float = 1e-3,
    rho: float = 3.0,
    t_max: float = 1.0,
    num_steps: int = 18,
    noise_scale: float = 0.1,
    noise_decay: float = 2.0,
    solver: Literal["euler", "heun"] = "euler",
    discretization: Literal["power", "edm"] = "power",
    seed: int = 0,
    use_true_scores: bool = False,
    verbose: bool = False,
) -> tuple[Batch, list[np.ndarray]]:
    """Evaluate the SE3Diffuser by performing a reverse sampling process.
    Args:
        denoiser: Instance of SE3Diffuser to use for sampling.
        batch: Batch of data containing the initial states and other necessary information.
        num_steps: Number of steps in the reverse sampling process.
        solver: Solver type for the ODE. Choices are:
            - euler (1x inference)
            - heun (2x inference)
            defaults to "euler".
        rho: Exponent for the time step discretization.
        noise_scale: Scale of noise to apply during the reverse sampling process.
        t_min: Minimum time step for the reverse sampling process.
        t_max: Maximum time step for the reverse sampling process.
        seed: Random seed for reproducibility.
        use_true_scores: If True, uses true scores for the reverse step instead of predicted scores from the denoiser.
    Returns:
        A tuple containing the updated batch and a list of positions at each step of the reverse sampling process.
    Raises:
        AssertionError: If noise_scale is not in the range [0, 1].
    """

    assert noise_scale >= 0, "Noise scale must be non-negative."
    assert noise_scale <= 1, "Noise scale must be less than or equal to 1."
    if isinstance(seed, int):
        assert seed >= 0, "Seed must be non-negative."

    # For each seed clone the batch and repeat the data items!
    # TODO. Currently outside loop.

    # Seed everything for reproducibility
    seed_everything(seed, workers=True, verbose=False)
    batch = denoiser._prepare_batch(batch)

    # Note this could be a random conformer in 3D space -> Typically we do this in inference.
    pos_0, T_0, R_0, num_fragments = denoiser._get_initial_states(batch)
    T_T, R_T = denoiser.diffuser.sample_ref(torch.sum(num_fragments), batch.x.device)

    # Update roto-translations to ambient space (complex)
    pos_T = denoiser._apply_transformations(
        pos_0=pos_0,
        batch=batch,
        T0=T_0,
        R0=R_0,
        T_t=T_T,
        R_t=R_T,
    )

    step_indices = torch.arange(num_steps, device=batch.x.device)  # [N]
    if discretization == "power":
        timesteps = torch.linspace(t_max, t_min, num_steps, device=batch.x.device) ** rho
    elif discretization == "edm":
        timesteps = (
            t_max ** (1 / rho) + step_indices / (num_steps - 1) * (t_min ** (1 / rho) - t_max ** (1 / rho))
        ) ** rho
    else:
        raise ValueError(f"Unknown discretization {discretization}. Choose 'power' or 'edm'.")

    # Initialize the denoiser with the initial states
    batch = denoiser._update_batch(
        batch=batch,
        pos_0=pos_0,
        pos_t=pos_T,
    )
    pos_t = pos_T
    T_t = T_T
    R_t = R_T

    if verbose:
        print(
            f"Using {num_steps} steps for reverse sampling with: \n \
            seed={seed} \n \
            rho={rho} \n \
            solver={solver} \n \
            noise_scale={noise_scale} \n \
            t_min={t_min} \n \
            "
        )

    @torch.no_grad
    def _reverse_step(
        batch: Batch,
        t: float,
        R_t: torch.Tensor,
        T_t: torch.Tensor,
        R_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]
        # Create discretization & Get time from discretization
        lig_pseudoforces, forces_idxs = denoiser._compute_forces(
            batch=batch,
            t=torch.tensor([t] * batch.num_graphs, device=pos_t.device),  # [B]
        )  # [B x F x A, 3], [B x F x A]
        # Linear mechanics: mass, inertia, force & torque
        force_per_fragment, torque_per_fragment, frag_mass, frag_inertia_t = denoiser._compute_fragment_dynamics(
            batch=batch,
            R_t=R_t,  # [B x F, 3, 3]
            T_t=T_t,  # [B x F, 3]
            # NOTE R0 only exists as safety during training for I_t calc. not required strictly (IRL)
            R0=R_0,  # [B x F, 3, 3]
            lig_forces=lig_pseudoforces,
            forces_idxs=forces_idxs,
        )  # [B x F, 3], [B x F, 3], [B,F], [B x F, 3, 3]

        # Compute total scaled forces/torques and predict updates (Newton-Maruyama)
        fragment_updates = denoiser._predict_fragment_updates(
            force_per_fragment=force_per_fragment,
            torque_per_fragment=torque_per_fragment,
            frag_mass=frag_mass,
            frag_inertia_t=frag_inertia_t,
            # t_batch=t_batch,
        )  # [B x F, 3], [B x F, 3, 3]

        # Compute [R3, so3] scores
        pred_scores = denoiser._compute_scores({"R_t": R_t, "T_t": T_t}, fragment_updates, t_batch)
        return pred_scores

    all_pos = [pos_t.cpu().numpy()]
    all_edges = [
        {
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "edge_entity": batch.edge_entity,
        }
    ]
    all_losses = []
    # Quadratic noise scaling reduction for reverse sampling
    noise_scales = torch.linspace(noise_scale ** (1 / noise_decay), 0.0, num_steps, device=batch.x.device) ** (
        noise_decay
    )  # [N]
    # Iterate across timesteps in reverse order
    for i, t in tqdm(enumerate(timesteps[:-1])):
        dt = timesteps[i] - timesteps[i + 1]
        noise_scale = noise_scales[i]  # [1]
        t = torch.tensor(t, device=batch.x.device)  # [1]
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]

        true_scores = denoiser._compute_true_scores(
            T0=T_0,
            R0=R_0,
            Tt=T_t,
            Rt=R_t,
            t_batch=t_batch,
        )

        grad_T_t = true_scores["true_T_score"]
        grad_R_t = true_scores["true_R_score"]
        # Use True or Predicted scores
        if not use_true_scores:
            # Deepcopy because reverse step modifies the batch in-place (removes masked edges)
            pred_scores = _reverse_step(deepcopy(batch), t, R_t=R_t, T_t=T_t, R_0=R_0)
            grad_T_p = pred_scores["pred_T_score"]
            grad_R_p = pred_scores["pred_R_score"]
        else:
            grad_T_p = grad_T_t
            grad_R_p = grad_R_t

        # Log the losses
        r3_scaling, so3_scaling = denoiser._get_scalings(t_batch)  # [B x F], [B x F]
        losses: dict[str, torch.Tensor] = denoiser.compute_losses(
            {
                "T_score_scaling": r3_scaling,  # [B x F]
                "R_score_scaling": so3_scaling,  # [B x F]
                "pred_T_score": grad_T_p,  # [B x F, 3]
                "pred_R_score": grad_R_p,  # [B x F, 3, 3]
                "true_T_score": grad_T_t,  # [B x F, 3]
                "true_R_score": grad_R_t,  # [B x F, 3, 3]
                "T_0": T_0,  # [B x F, 3]
                "T_0_hat": T_0,
                "R_0_hat": R_0,  # [B x F, 3, 3]
                "R_0": R_0,  # [B x F, 3, 3]
                "t_batch": t_batch,  # [B x F]
            }
        )
        # Reverse step (use discretization & ODE solver)
        T_next, R_next = denoiser.diffuser.reverse(
            trans_t=T_t,
            R_t=R_t,
            trans_score=grad_T_p,
            rot_score=grad_R_p,
            noise_scale=noise_scale,  # No noise in reverse step
            t=t,
            dt=dt,
        )

        # Update positions according to transformations from reverse step.
        pos_t = denoiser._apply_transformations(
            batch=batch,
            # Refererence
            pos_0=pos_t,
            T0=T_t,
            R0=R_t,
            # Transformation
            T_t=T_next,
            R_t=R_next,
        )
        # Update roto-translations with reverse kinematics for next step.
        T_t = T_next
        R_t = R_next
        # Update batch with new positions (pos_t) and remove prev local interactions
        batch = denoiser._update_batch(
            batch=batch,
            pos_0=pos_0,
            pos_t=pos_t,
        )

        all_pos.append(pos_t.cpu().numpy())
        all_edges.append(
            {
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "edge_entity": batch.edge_entity,
            }
        )
        all_losses.append(losses)

        # TODO Heun's method
        #  Will require us to look at derivatives for T and R and average the Func Evaluations at t, t-t'
        #  Remember last step must be Euler so NFE = 2 * (N - 1) + 1
        #  Truncation error: O(N) -> O(N**3) which might allow us to do less steps (but who cares tho).

    is_lig = torch.where(batch.frag_idx_map != -1)[0]
    ref_lig_pos = batch.ref_pos
    pred_lig_pos = batch.pos_t * HPARAMS.general.dimensional_scale + batch.pocket_com.repeat_interleave(
        torch.bincount(batch.batch), dim=0
    )
    dev = (ref_lig_pos[is_lig] - pred_lig_pos[is_lig]).norm(dim=-1)
    # print(f"Average Deviation {dev.mean()}")
    return batch, all_pos, all_edges, all_losses


# Keeping for versioning JIC.
def sampler(
    denoiser: SE3Diffuser,
    batch: Batch,
    t_min: float = 1e-3,
    rho: float = 3.0,
    t_max: float = 1.0,
    num_steps: int = 18,
    noise_scale: float = 0.1,
    noise_decay: float = 2.0,
    solver: Literal["euler", "heun"] = "euler",
    discretization: Literal["power", "edm"] = "power",
    use_true_scores: bool = False,
    verbose: bool = False,
) -> tuple[Batch, list[np.ndarray]]:
    """Evaluate the SE3Diffuser by performing a reverse sampling process.
    Args:
        denoiser: Instance of SE3Diffuser to use for sampling.
        batch: Batch of data containing the initial states and other necessary information.
        num_steps: Number of steps in the reverse sampling process.
        solver: Solver type for the ODE. Choices are:
            - euler (1x inference)
            - heun (2x inference)
            defaults to "euler".
        rho: Exponent for the time step discretization.
        noise_scale: Scale of noise to apply during the reverse sampling process.
        t_min: Minimum time step for the reverse sampling process.
        t_max: Maximum time step for the reverse sampling process.
        seed: Random seed for reproducibility.
        use_true_scores: If True, uses true scores for the reverse step instead of predicted scores from the denoiser.
    Returns:
        A tuple containing the updated batch and a list of positions at each step of the reverse sampling process.
    Raises:
        AssertionError: If noise_scale is not in the range [0, 1].
    """

    assert noise_scale >= 0, "Noise scale must be non-negative."
    assert noise_scale <= 1, "Noise scale must be less than or equal to 1."

    # TODO For each seed clone the batch and repeat the data items!
    batch = denoiser._prepare_batch(batch)

    # Note this could be a random conformer in 3D space -> Typically we do this in inference.
    pos_0, T_0, R_0, num_fragments = denoiser._get_initial_states(batch)
    T_T, R_T = denoiser.diffuser.sample_ref(torch.sum(num_fragments), batch.x.device)

    # Update roto-translations to ambient space (complex)
    pos_T = denoiser._apply_transformations(
        pos_0=pos_0,
        batch=batch,
        T0=T_0,
        R0=R_0,
        T_t=T_T,
        R_t=R_T,
    )

    step_indices = torch.arange(num_steps, device=batch.x.device)  # [N]
    if discretization == "power":
        timesteps = torch.linspace(t_max, t_min, num_steps, device=batch.x.device) ** rho
    elif discretization == "edm":
        timesteps = (
            t_max ** (1 / rho) + step_indices / (num_steps - 1) * (t_min ** (1 / rho) - t_max ** (1 / rho))
        ) ** rho
    else:
        raise ValueError(f"Unknown discretization {discretization}. Choose 'power' or 'edm'.")

    # Initialize the denoiser with the initial states
    batch = denoiser._update_batch(
        batch=batch,
        pos_0=pos_0,
        pos_t=pos_T,
    )
    pos_t = pos_T
    T_t = T_T
    R_t = R_T

    if verbose:
        print(
            f"Using {num_steps} steps for reverse sampling with: \n \
            rho={rho} \n \
            solver={solver} \n \
            noise_scale={noise_scale} \n \
            t_min={t_min} \n \
            "
        )

    @torch.no_grad
    def _reverse_step(
        batch: Batch,
        t: float,
        R_t: torch.Tensor,
        T_t: torch.Tensor,
        R_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]
        # Create discretization & Get time from discretization
        lig_pseudoforces, forces_idxs = denoiser._compute_forces(
            batch=batch,
            t=torch.tensor([t] * batch.num_graphs, device=pos_t.device),  # [B]
        )  # [B x F x A, 3], [B x F x A]
        # Linear mechanics: mass, inertia, force & torque
        force_per_fragment, torque_per_fragment, frag_mass, frag_inertia_t = denoiser._compute_fragment_dynamics(
            batch=batch,
            R_t=R_t,  # [B x F, 3, 3]
            T_t=T_t,  # [B x F, 3]
            # NOTE R0 only exists as safety during training for I_t calc. not required strictly (IRL)
            R0=R_0,  # [B x F, 3, 3]
            lig_forces=lig_pseudoforces,
            forces_idxs=forces_idxs,
        )  # [B x F, 3], [B x F, 3], [B,F], [B x F, 3, 3]

        # Compute total scaled forces/torques and predict updates (Newton-Maruyama)
        fragment_updates = denoiser._predict_fragment_updates(
            force_per_fragment=force_per_fragment,
            torque_per_fragment=torque_per_fragment,
            frag_mass=frag_mass,
            frag_inertia_t=frag_inertia_t,
            # t_batch=t_batch,
        )  # [B x F, 3], [B x F, 3, 3]

        # Compute [R3, so3] scores
        pred_scores = denoiser._compute_scores({"R_t": R_t, "T_t": T_t}, fragment_updates, t_batch)
        return pred_scores

    all_losses = []
    # Quadratic noise scaling reduction for reverse sampling
    noise_scales = torch.linspace(noise_scale ** (1 / noise_decay), 0.0, num_steps, device=batch.x.device) ** (
        noise_decay
    )  # [N]

    # Get ligand indices
    is_lig = torch.where(batch.frag_idx_map != -1)[0]
    # Initialize with Stationary sample

    all_pos = [pos_t[is_lig]]
    # Iterate across timesteps in reverse order
    for i, t in tqdm(enumerate(timesteps[:-1])):
        dt = timesteps[i] - timesteps[i + 1]
        noise_scale = noise_scales[i]  # [1]
        t = torch.tensor(t, device=batch.x.device)  # [1]
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]

        true_scores = denoiser._compute_true_scores(
            T0=T_0,
            R0=R_0,
            Tt=T_t,
            Rt=R_t,
            t_batch=t_batch,
        )

        grad_T_t = true_scores["true_T_score"]
        grad_R_t = true_scores["true_R_score"]
        # Use True or Predicted scores
        if not use_true_scores:
            # Deepcopy because reverse step modifies the batch in-place (removes masked edges)
            pred_scores = _reverse_step(deepcopy(batch), t, R_t=R_t, T_t=T_t, R_0=R_0)
            grad_T_p = pred_scores["pred_T_score"]
            grad_R_p = pred_scores["pred_R_score"]
        else:
            grad_T_p = grad_T_t
            grad_R_p = grad_R_t

        # Log the losses
        r3_scaling, so3_scaling = denoiser._get_scalings(t_batch)  # [B x F], [B x F]
        losses: dict[str, torch.Tensor] = denoiser.compute_losses(
            {
                "T_score_scaling": r3_scaling,  # [B x F]
                "R_score_scaling": so3_scaling,  # [B x F]
                "pred_T_score": grad_T_p,  # [B x F, 3]
                "pred_R_score": grad_R_p,  # [B x F, 3, 3]
                "true_T_score": grad_T_t,  # [B x F, 3]
                "true_R_score": grad_R_t,  # [B x F, 3, 3]
                "T_0": T_0,  # [B x F, 3]
                "T_0_hat": T_0,
                "R_0_hat": R_0,  # [B x F, 3, 3]
                "R_0": R_0,  # [B x F, 3, 3]
                "t_batch": t_batch,  # [B x F]
            }
        )
        # Reverse step (use discretization & ODE solver)
        T_next, R_next = denoiser.diffuser.reverse(
            trans_t=T_t,
            R_t=R_t,
            trans_score=grad_T_p,
            rot_score=grad_R_p,
            noise_scale=noise_scale,  # No noise in reverse step
            t=t,
            dt=dt,
        )

        # Update positions according to transformations from reverse step.
        pos_t = denoiser._apply_transformations(
            batch=batch,
            # Refererence
            pos_0=pos_t,
            T0=T_t,
            R0=R_t,
            # Transformation
            T_t=T_next,
            R_t=R_next,
        )
        # Update roto-translations with reverse kinematics for next step.
        T_t = T_next
        R_t = R_next
        # Update batch with new positions (pos_t) and remove prev local interactions
        batch = denoiser._update_batch(
            batch=batch,
            pos_0=pos_0,
            pos_t=pos_t,
        )

        all_pos.append(pos_t[is_lig])
        all_losses.append(losses)

        # TODO Heun's method
        #  Will require us to look at derivatives for T and R and average the Func Evaluations at t, t-t'
        #  Remember last step must be Euler so NFE = 2 * (N - 1) + 1
        #  Truncation error: O(N) -> O(N**3) which might allow us to do less steps (but who cares tho).

    ref_lig_pos = batch.ref_pos
    pred_lig_pos = batch.pos_t * HPARAMS.general.dimensional_scale + batch.pocket_com.repeat_interleave(
        torch.bincount(batch.batch), dim=0
    )
    dev = (ref_lig_pos[is_lig] - pred_lig_pos[is_lig]).norm(dim=-1)
    # print(f"Average Deviation {dev.mean()}")
    return batch, torch.stack(all_pos, dim = 0), all_losses
