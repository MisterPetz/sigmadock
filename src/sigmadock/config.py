import os
import re
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import torch
import yaml
from omegaconf import MISSING

from sigmadock.oracle import HPARAMS, ROOT_DIR
from sigmadock.torch_utils.dist import reliable_world_size


def update_config_from_args(config: dataclass, args: dataclass) -> dataclass:
    if not is_dataclass(config):
        raise TypeError("Expected a dataclass for config")

    sc_fields = {f.name for f in fields(config)}
    arg_dict = vars(args)

    updates = {k: v for k, v in arg_dict.items() if k in sc_fields}
    return replace(config, **updates)


@dataclass
class StructuralConfig:
    """
    Configuration class for structural parameters (cheminformatics and graph construction).
    """

    # Cheminformatics
    pocket_com_cutoff: float = 6.0
    pocket_distance_cutoff: float = 5.0
    pocket_com_noise: float = 0.5
    pocket_distance_noise: float = 1.0
    lig_coordinate_distance_noise: float = 0
    prot_coordinate_distance_noise: float = 0.02  # Coordinate Jitter ~ Thermal Noise (High Frequency Vibrations)
    pocket_residue_outlier_factor: float = -1

    # TODO this should be a config option.
    # NOTE This is dangerous logic. Switch config to Hydra Config Store for safety.
    pocket_virtual_cutoff: float = getattr(HPARAMS.get_edge_spec("protein_v2v"), "r_max", -1)

    # Augmentation
    random_rotation: bool = False  # Whether to apply random rotation to the protein
    mirror_prob: float = 0.0  # Probability of mirroring the complex

    # Fragmentation
    alignment_tries: int = 3
    fragmentation_strategy: Literal["random", "random_all", "max", "largest", "smallest", "canonical"] = "random"
    alignment_rmsd_tolerance: float = 1.0
    alignment_energy_tolerance: float = 10.0
    ignore_triangulation: bool = False  # Ignore triangulation indexes in the dataset. Only for ablations.

    # Chemo Misc
    streamloading: bool = True
    keep_hetatoms: bool = False
    ignore_conjugated_torsion: bool = False

    # Graph Construction
    pb_check: bool = False  # PoseBusters Check in data generation
    include_protein_ligand_interactions: bool = False
    include_fragment_fragment_interactions: bool = False
    use_esm_embeddings: bool = False
    force_retry: bool = True  # Force retry on error in data generation. True is safer, False is faster.

    esm_embeddings_clip_range: Optional[tuple[float, float]] = None
    esm_embeddings_scaling_factor: float = 1
    atom_feature_dims: list[int] = (54, 5, 3, 4, 4, 7, 2, 2, 4, 3)
    edge_feature_dims: list[int] = (5, 2, 2, 4)
    average_degrees: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.average_degrees = (
            HPARAMS.all_degrees if self.include_protein_ligand_interactions else HPARAMS.global_degrees
        )


@dataclass
class TrainingConfig:
    """
    Configuration class for traininig SigmaDock.
    """

    # Architecture
    sphere_channels: int = 128  # Featurised Input Dimension data.x
    edge_channels: int = 32  # Featurised Edge Dimension data.edge_attr
    distance_expansion_dim: int = 32
    num_heads: int = 4
    num_layers: int = 6
    l_max_list: list[int] = (3,)
    m_max_list: list[int] = (2,)
    # Hidden layers
    attn_hidden_channels: int = 64
    attn_alpha_channels: int = 32
    attn_value_channels: int = 16
    ffn_hidden_channels: int = 128
    # Distance & Time Dimensions
    t_emb_dim: int = 32
    t_emb_type: Literal["sinusoidal", "fourier"] = "sinusoidal"
    t_emb_scale: float = 128.0  # Scale for the time embedding,
    smearing_type: Literal["gaussian", "symmetric-fourier", "fourier", "sigmoid"] = "fourier"
    radial_type: Literal["bessel", "gaussian"] = "bessel"
    rel_distance: bool = True  # Whether to use relative distance expansion for boundary conditions
    # Init last layer (force block) to output zero at init (equi-Variance with time)
    zero_init_last: bool = True

    # Rotation components (deprecated compatibility)
    rot_score_method: Literal["space", "score"] = "score"  # Either predict space: delta_R or score_R from the model
    rot_score_scaling: Literal["rms", "true"] | None = "true"  # How to scale the rotation score
    reverse_rotations: bool = False

    # Losses
    fragment_scaling: float = 1.0  # 1.0 does not scale, averages per mol. 0.0 scales each fragment same weight p.
    # Score losses
    trans_score_weight: float = 1
    rot_score_weight: float = 0.5
    # Data losses
    trans_data_weight: float = 0.0
    rot_data_weight: float = 0.0

    # Optimizer & Scheduler TODO add momentum and beta...
    attention_dropout: float = 0.3
    edge_dropout: float = 0.1
    weight_decay: float = 0.1
    optimizer_eps: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.999)  # AdamW betas (B2 reduced to 0.995 for stability)
    # Linear Warmup
    init_lr_start: float = 1e-8
    lr_warmup_frac: float = 1 / 16
    # Cycle Cosine Annealing with Decay
    num_lr_cycles: int = 1
    cycle_warmup_frac: float = 1 / 4
    max_lr_start: float = 1e-4
    min_lr_start: float = 1e-5
    max_lr_end: float = 1e-5
    min_lr_end: float = 1e-6
    grad_clip: float | None = 4.0

    # Data Loading
    max_epochs: int | None = 256  # Max epochs if max_steps is None
    max_steps: int | None = None
    val_check_interval: int | None = None  # Check validation every N steps
    batch_size: int = 64  # Batch size for training
    accum_grad_batches: int = 1  # Gradient accumulation steps
    num_workers: int | str | None = "auto"  # Number of workers for data loading
    cache_factor: int = 2
    cache_cycles: int = 8
    # NOTE this should not really exist if we are using a BIG dataset.
    dataset_augmentation_factor: int = 1  # Number of times to augment the dataset
    val_cycles: int = 4  # Number of validation cycles

    # Callbacks
    monitor_metric: str = "loss_val/total"
    early_stopping_patience: int | float = 1 / 4  # In Epochs / ratio of max epochs or max_stepss
    use_ema: bool = True
    ema_rampup_ratio: float = 1 / 8
    ema_halflife: float | int = 2  # Halflife in number of "len(dataset)" points (equiv to epochs)

    def __post_init__(self) -> None:
        # Ensure that the fragment scaling is between 0 and 1
        assert self.fragment_scaling > 0, "Fragment scaling must be greater than 0."
        assert self.fragment_scaling <= 1, "Fragment scaling must be less than or equal to 1."

        self.l_max_list = list(self.l_max_list)
        self.m_max_list = list(self.m_max_list)

        if self.num_workers == "auto":
            if torch.cuda.is_available():
                self.num_workers = os.cpu_count() - 1
            else:
                self.num_workers = os.cpu_count() // 2
        else:
            self.num_workers = int(float(self.num_workers))


@dataclass
class ExperimentConfig:
    _target_: str = MISSING
    name: str = MISSING
    dataset: Path = MISSING
    pdb_regex: str = MISSING
    sdf_regex: str = MISSING
    # Optional: regex to find a reference SDF in each complex folder for pocket definition (e.g. "reference\\.sdf")
    ref_sdf_regex: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, Path):
            self.dataset = Path(self.dataset)

        if not self.dataset.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset}")

        # Ensure the dataset is a directory
        if not self.dataset.is_dir():
            raise ValueError(f"Dataset path is not a directory: {self.dataset}")

        # Ensure the dataset has a valid name
        if self.name == "":
            raise ValueError("Experiment name cannot be empty")

        # Ensure the regex patterns are valid
        try:
            re.compile(self.pdb_regex)
            re.compile(self.sdf_regex)
            if self.ref_sdf_regex is not None:
                re.compile(self.ref_sdf_regex)
        except re.error as e:
            raise ValueError(f"Invalid regex: {e}") from e


@dataclass
class RunConfig(TrainingConfig, StructuralConfig):
    # Experiment Config
    experiment: str = "sigmadock"
    exp_dir: str | Path = ROOT_DIR / "experiments/"
    data_dir: Path = ROOT_DIR / "data"
    train_exps: list[str] = field(default_factory=lambda: ["pdbbind-general", "pdbbind-refined"])
    val_exps: list[str] = field(default_factory=lambda: ["pdbbind-core"])
    test_exps: list[str] = field(default_factory=lambda: ["astex", "posebusters"])
    resume_from_checkpoint: bool | str | Path = False

    # Trainer
    accelerator: str = "auto"  # "gpu", "cpu", "mps", "auto" #
    strategy: str = "auto"  # "ddp", "dp", "fsdp", "auto" #
    devices: int | str = "auto"  # Number of devices to use
    precision: str | None = None  # "16", "16-mixed", "32", "64", "64-mixed"
    cuda_precision: Literal["highest", "high", "medium"] = "highest"
    compile: bool = False  # Whether to compile the model with torch.compile

    # Misc
    debug: bool = False
    seed: int = 0  # Random seed
    deterministic: bool = True  # This makes it a bit slower but reproducible
    offline_run: bool = False
    log_model: bool = True
    world_size: int = 1  # This is set automatically, do not change

    def __post_init__(self) -> None:
        TrainingConfig.__post_init__(self)
        StructuralConfig.__post_init__(self)
        # Note scaling batch size according to the number of GPUs
        # NOTE this alternative is more correct for non-slurm: world_size = get_world_size()
        world_size = reliable_world_size()
        self.world_size = world_size
        if world_size > 1:
            print(f"Scaling batch size by {1 / world_size} for distributed training.")
            self.batch_size = self.batch_size // world_size
        else:
            print("Using batch size as is for single GPU training.")
        # Init experiments using ExperimentConfig?
        # self.train_dir = Path(self.data_dir) / "posebusters_paper/posebusters_benchmark_set/"
        # self.val_dir = Path(self.data_dir) / "posebusters_paper/astex_diverse_set/"


@dataclass
class EnergyMinimisationConfig:
    """
    Configuration class for post-processing energy minimisation.
    """

    minimise_energy: bool = True
    tolerance: float = 0.01
    allow_undefined_stereo: bool = True
    add_solvent: bool = False
    platform_name: str = "fastest"
    device_str: str = "6"


# ----------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------


def parse_args_from_configs(configs: list) -> Namespace:
    """
    Automatically creates a command-line parser by inspecting a list of dataclasses.
    """
    parser = ArgumentParser()

    # Keep track of arguments to prevent duplicates
    added_args = set()

    for config_class in configs:
        # Loop through all the fields in the dataclass (e.g., 'batch_size', 'num_layers')
        for field in fields(config_class):  # noqa
            # Skip if we've already added this argument from another config class
            if field.name in added_args:
                continue

            # For boolean flags, use a more robust action
            if field.type == bool:  # noqa
                parser.add_argument(
                    f"--{field.name}",
                    action="store_true",  # Or use argparse.BooleanOptionalAction in Python 3.9+
                    default=None,
                    help=f"Override {field.name}. Default: {field.default}",
                )
            else:
                # For other types, add the argument normally
                parser.add_argument(
                    f"--{field.name}",
                    type=field.type,
                    default=None,
                    help=f"Override {field.name}. Default: {field.default}",
                )

            added_args.add(field.name)

    return parser.parse_args()


def parse_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser()
    # Config file (YAML); applied before other CLI overrides
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML training config. CLI args override config file values.",
    )
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode.",
    )
    # Compile
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for the model.",
    )
    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    # Checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=lambda s: s if s.lower() != "false" else None,
        default=None,
        help="Path to the checkpoint to resume from, or 'False' to not resume.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--train_exps",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated experiment names for training (e.g. 'pdbbind-core,astex')",
    )
    parser.add_argument(
        "--val_exps",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated experiment names for validation",
    )
    parser.add_argument(
        "--test_exps",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated experiment names for testing",
    )
    # Val cycles
    parser.add_argument(
        "--val_cycles",
        type=int,
        default=None,
        help="Number of validation cycles to run.",
    )
    # Epochs
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs for training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps for training. Overrides max_epochs if set.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=None,
        help="Check validation every N steps.",
    )
    # Early stopping
    parser.add_argument(
        "--early_stopping_patience",
        type=float,
        default=None,
        help="Patience for early stopping (in epochs or fraction of max epochs).",
    )
    # Learning rates
    parser.add_argument(
        "--init_lr_start",
        type=float,
        default=None,
        help="Initial learning rate at the start of training.",
    )
    parser.add_argument(
        "--max_lr_start",
        type=float,
        default=None,
        help="Maximum learning rate at the start of the cycle.",
    )
    parser.add_argument(
        "--max_lr_end",
        type=float,
        default=None,
        help="Maximum learning rate at the end of the cycle.",
    )
    parser.add_argument(
        "--min_lr_start",
        type=float,
        default=None,
        help="Minimum learning rate at the start of the cycle.",
    )
    parser.add_argument(
        "--min_lr_end",
        type=float,
        default=None,
        help="Minimum learning rate at the end of the cycle.",
    )
    parser.add_argument(
        "--num_lr_cycles",
        type=int,
        default=None,
        help="Number of learning rate cycles to run.",
    )
    parser.add_argument(
        "--cycle_warmup_frac",
        type=float,
        default=None,
        help="Fraction of the cycle to warm up the learning rate.",
    )
    parser.add_argument(
        "--lr_warmup_frac",
        type=float,
        default=None,
        help="Fraction of the total training epochs to warm up the learning rate.",
    )
    # Augment
    parser.add_argument(
        "--force_retry",
        action="store_true",
        default=None,
        help="Force retry on error in data generation. True is safer, False is faster.",
    )
    parser.add_argument(
        "--fragmentation_strategy",
        type=str,
        choices=["random", "random_all", "max", "largest", "smallest", "canonical"],
        default=None,
        help="Strategy for fragmentation.",
    )
    parser.add_argument(
        "--alignment_rmsd_tolerance",
        type=float,
        default=None,
        help="RMSD tolerance for alignment during fragmentation.",
    )
    parser.add_argument(
        "--alignment_energy_tolerance",
        type=float,
        default=None,
        help="Energy tolerance for alignment during fragmentation.",
    )
    # Rotation
    parser.add_argument(
        "--rot_score_method",
        type=str,
        choices=["space", "score"],
        default=None,
        help="Method for rotation score: 'space' for delta_R, 'score' for score_R.",
    )
    parser.add_argument(
        "--rot_score_scaling",
        type=str,
        choices=["rms", "true", None],
        default=None,
        help="How to scale the rotation score. 'rms' for RMS scaling, 'true' for true scaling, None for no scaling.",
    )
    parser.add_argument(
        "--reverse_rotations",
        action="store_true",
        default=None,
        help="Reverse rotations in the denoiser.",
    )
    # Random rotation
    parser.add_argument(
        "--random_rotation",
        action="store_true",
        default=None,
        help="Whether to apply random rotation to the protein.",
    )
    parser.add_argument(
        "--mirror_prob",
        type=float,
        default=None,
        help="Probability of mirroring the complex.",
    )
    # Distance Noises and Cutoffs
    parser.add_argument(
        "--pocket_com_cutoff",
        type=float,
        default=None,
        help="Cutoff for the center of mass distance in the pocket.",
    )
    parser.add_argument(
        "--pocket_distance_cutoff",
        type=float,
        default=None,
        help="Cutoff for the distance in the pocket.",
    )
    parser.add_argument(
        "--pocket_com_noise",
        type=float,
        default=None,
        help="Noise to add to the center of mass distance in the pocket.",
    )
    parser.add_argument(
        "--pocket_distance_noise",
        type=float,
        default=None,
        help="Noise to add to the distance in the pocket.",
    )
    parser.add_argument(
        "--lig_coordinate_distance_noise",
        type=float,
        default=None,
        help="Noise to add to the ligand coordinate distance.",
    )
    parser.add_argument(
        "--prot_coordinate_distance_noise",
        type=float,
        default=None,
        help="Noise to add to the protein coordinate distance.",
    )
    parser.add_argument(
        "--pocket_residue_outlier_factor",
        type=float,
        default=None,
        help="Factor to determine outliers in the pocket residues.",
    )
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training.",
    )
    # Cache & Size
    parser.add_argument(
        "--cache_factor",
        type=int,
        default=None,
        help="Factor to increase the cache size for the dataset.",
    )
    parser.add_argument(
        "--cache_cycles",
        type=int,
        default=None,
        help="Number of cycles to cache the dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=str,
        default=None,
        help="Number of workers for data loading.",
    )
    # Architecture
    parser.add_argument(
        "--sphere_channels",
        type=int,
        default=None,
        help="Number of channels for the sphere input.",
    )
    parser.add_argument(
        "--edge_channels",
        type=int,
        default=None,
        help="Number of channels for the edge features.",
    )
    parser.add_argument(
        "--distance_expansion_dim",
        type=int,
        default=None,
        help="Number of channels for the edge distance features.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of layers in the model.",
    )
    # Spherical Harmonics
    parser.add_argument(
        "--l_max_list",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="List of l_max values for spherical harmonics.",
    )
    parser.add_argument(
        "--m_max_list",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="List of m_max values for spherical harmonics.",
    )
    # Zero init
    parser.add_argument(
        "--zero_init_last",
        action="store_true",
        default=None,
        help="Whether to zero initialize the last layer (force block) to output zero at init.",
    )
    # Grad clip
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Gradient clipping value.",
    )
    # Ema halflife
    parser.add_argument(
        "--ema_halflife",
        type=float,
        default=None,
        help="Halflife for the Exponential Moving Average (EMA) in number of 'len(dataset)' points.",
    )
    # Weight Decay
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay for the optimizer.",
    )
    # Optimizer
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=None,
        help="Epsilon value for the optimizer.",
    )
    parser.add_argument(
        "--betas",
        type=lambda s: tuple(float(x) for x in s.split(",")),
        default=None,
        help="Betas for the optimizer (e.g., '0.9,0.999').",
    )
    # Dropouts
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=None,
        help="Dropout rate for attention layers.",
    )
    parser.add_argument(
        "--edge_dropout",
        type=float,
        default=None,
        help="Dropout rate for edge features.",
    )
    # EMA
    parser.add_argument(
        "--use_ema",
        type=bool,
        default=None,
        help="Use Exponential Moving Average for model weights.",
    )
    # Smearing
    parser.add_argument(
        "--smearing_type",
        type=str,
        choices=["gaussian", "symmetric-fourier", "fourier", "sigmoid"],
        default=None,
        help="Type of smearing to use for distance expansion.",
    )
    # Radial Basis
    parser.add_argument(
        "--radial_type",
        type=str,
        choices=["bessel", "gaussian"],
        default=None,
        help="Type of radial basis function to use.",
    )
    # Rel distances
    parser.add_argument(
        "--rel_distance",
        type=bool,
        default=True,
        help="Use relative distance expansion for boundary conditions.",
    )
    # Ignore triangulation
    parser.add_argument(
        "--ignore_triangulation",
        action="store_true",
        default=None,
        help="Ignore triangulation indexes in the dataset. This is only intended for Ablations.",
    )
    # Interactions
    parser.add_argument(
        "--include_protein_ligand_interactions",
        action="store_true",
        default=None,
        help="Include interactions in the graph construction.",
    )
    parser.add_argument(
        "--include_fragment_fragment_interactions",
        action="store_true",
        default=None,
        help="Include fragment-fragment interactions in the graph construction.",
    )
    # Fragmentation
    parser.add_argument(
        "--alignment_tries",
        type=int,
        default=None,
        help="Number of tries for alignment during fragmentation.",
    )
    # Conjugation
    parser.add_argument(
        "--ignore_conjugated_torsion",
        action="store_true",
        default=None,
        help="Ignore conjugated torsions during fragmentation.",
    )
    parser.add_argument(
        "--pb_check",
        action="store_true",
        default=None,
        help="Enable PoseBusters check in data generation.",
    )
    # ESM embeddings
    parser.add_argument(
        "--use_esm_embeddings",
        action="store_true",
        default=None,
        help="Use ESM embeddings for atom features.",
    )
    # Time scaling
    parser.add_argument(
        "--t_emb_type",
        type=str,
        choices=["sinusoidal", "fourier"],
        default=None,
        help="Type of time embedding to use.",
    )
    parser.add_argument(
        "--t_emb_scale",
        type=float,
        default=None,
        help="Scale for the time embedding.",
    )
    parser.add_argument(
        "--t_emb_dim",
        type=int,
        default=None,
        help="Dimension of the time embedding.",
    )
    # Loss weighting
    parser.add_argument(
        "--fragment_scaling",
        type=float,
        default=None,
        help="Fragment scaling factor.",
    )
    # Translation & Rotation loss weights!
    parser.add_argument(
        "--trans_score_weight",
        type=float,
        default=None,
        help="Weight for translation score loss.",
    )
    parser.add_argument(
        "--rot_score_weight",
        type=float,
        default=None,
        help="Weight for rotation score loss.",
    )
    # Translation & Rotation data weights
    parser.add_argument(
        "--trans_data_weight",
        type=float,
        default=None,
        help="Weight for translation data loss.",
    )
    parser.add_argument(
        "--rot_data_weight",
        type=float,
        default=None,
        help="Weight for rotation data loss.",
    )
    # Hardware
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Accelerator to use (e.g., 'gpu', 'cpu', 'mps').",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Distributed strategy to use (e.g., 'ddp', 'dp').",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Number of devices to use (e.g., 'auto', '1', '2').",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision to use (e.g., '16', 'mixed', '32').",
    )
    parser.add_argument(
        "--cuda_precision",
        type=str,
        default=None,
        help="CUDA precision to use (e.g., 'highest', 'high', 'medium').",
    )
    # Offlien run
    parser.add_argument(
        "--offline_run",
        action="store_true",
        default=None,
        help="Run in offline mode (no logging, no checkpoints).",
    )
    return parser.parse_args()


def get_exp_dir_from_ckpt(ckpt_path: str) -> Path:
    """
    Given a checkpoint path, derive the experiment directory (EXP_DIR).
    """
    checkpoint_dir = Path(ckpt_path).parent
    return checkpoint_dir.parent  # The parent directory is where the experiment folder should be


def get_exp_dir(args: RunConfig) -> Path:
    """
    Dynamically generates the experiment directory based on parameters and timestamp.
    """
    timestamp = time.strftime("%m-%d_%H-%M-%S")
    subdir_name = f"{args.seed}-{timestamp}"
    # If exists, need a versioning system
    exp_dir = args.exp_dir / args.experiment / subdir_name
    version = 1
    while exp_dir.exists():
        version += 1
        subdir_name = f"{args.seed}-{timestamp}-v{version}"
        exp_dir = args.exp_dir / args.experiment / subdir_name
    return args.exp_dir / args.experiment / subdir_name


def get_experiment_config(experiment: str, root_dir: str | Path) -> ExperimentConfig:
    """
    Returns the experiment configuration.
    """
    assert root_dir.exists(), f"Root directory does not exist: {root_dir}"
    conf_dir = Path(__file__).parent.parent.parent / "conf"
    conf_path = conf_dir / "experiments" / f"{experiment}.yaml"
    if not conf_path.exists():
        raise ValueError(f"Experiment config not found: {conf_path}")

    with open(conf_path) as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Experiment config is empty: {conf_path}")

    if root_dir is not None:
        config["dataset"] = Path(root_dir) / config["dataset"]
    return ExperimentConfig(
        name=experiment,
        dataset=config["dataset"],
        pdb_regex=config["pdb_regex"],
        sdf_regex=config["sdf_regex"],
        ref_sdf_regex=config.get("ref_sdf_regex"),
    )
