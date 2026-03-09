import copy
import math
import random
from itertools import cycle, islice
import matplotlib.colors as mcolors
import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Geometry import Point3D


def view_mol(mol: Chem.Mol, style: str = "stick") -> py3Dmol.view:
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mb, "mol")
    view.setStyle({style: {}})
    view.zoomTo()
    return view


def add_marker_to_view(view: py3Dmol.view, center: np.ndarray, marker_options: dict[str, str] | None = None) -> None:
    """
    Adds a marker (a sphere by default) to the given py3Dmol view.

    Args:
        view: The py3Dmol view to add the marker to.
        center: A (3,) NumPy array representing the (x, y, z) center of the marker.
        marker_options: A dictionary of options to customize the marker (e.g., radius, color, opacity).
            Defaults to a sphere of radius 1.5, color "magenta", opacity 0.7.
    """
    # Default options
    options = {"marker_type": "sphere", "radius": 1.5, "color": "magenta", "opacity": 0.7}
    if marker_options:
        options.update(marker_options)

    # Only sphere is implemented here
    if options["marker_type"] == "sphere":
        view.addSphere(
            {
                "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "radius": options["radius"],
                "color": options["color"],
                "opacity": options["opacity"],
            }
        )


def view_complex(
    protein_mol: Chem.Mol,
    ligand_mol: Chem.Mol,
    com: np.ndarray | None = None,
    protein_style: str = "cartoon",
    ligand_style: str = "stick",
    ligand_color: str = "red",
    width: int = 800,
    height: int = 500,
) -> py3Dmol.view:
    # Ensure 3D coordinates exist
    assert protein_mol.GetConformer().Is3D()
    # AllChem.EmbedMolecule(protein_mol)
    assert ligand_mol.GetConformer().Is3D()
    # AllChem.EmbedMolecule(ligand_mol)

    protein_block = Chem.MolToPDBBlock(protein_mol)
    ligand_block = Chem.MolToPDBBlock(ligand_mol)

    view = py3Dmol.view(width=width, height=height)

    # Add protein model
    view.addModel(protein_block, "pdb")

    chain_color_list = ["red", "blue", "green", "yellow", "orange", "purple"]
    # Remove lig color from the list
    chain_color_list.remove(ligand_color)

    # Color the protein by chain
    chain_colors = {}
    for atom in protein_mol.GetAtoms():
        chain_id = atom.GetPDBResidueInfo().GetChainId()
        if chain_id not in chain_colors:
            # Random color for each chain
            chain_colors[chain_id] = np.random.choice(chain_color_list)
            chain_color_list.remove(chain_colors[chain_id])

    # Set style and color by chain
    for chain_id, color in chain_colors.items():
        view.setStyle(
            {"chain": chain_id},
            {protein_style: {"color": color}},
        )

    # Add ligand model with custom color
    view.addModel(ligand_block, "pdb")
    view.setStyle({"model": 1}, {ligand_style: {"color": ligand_color}})

    # Optionally add a marker for the CoM if provided
    if com is not None:
        add_marker_to_view(view, com)

    view.zoomTo()
    return view


def remove_dummies_and_sanitize(mol: Chem.Mol) -> Chem.Mol:
    """
    Removes dummy atoms (atomic number 0) from the molecule, preserving
    3D conformer positions as best as possible, and sanitizes without kekulization.
    Logs the process so you can inspect the atom counts.
    """
    # Deep copy to avoid modifying the original molecule
    new_mol = copy.deepcopy(mol)

    # Check for and store conformer positions (if available)
    has_conf = new_mol.GetNumConformers() > 0
    if has_conf:
        conf = new_mol.GetConformer()
        positions = [conf.GetAtomPosition(i) for i in range(new_mol.GetNumAtoms())]
    else:
        positions = None

    # Create an editable molecule
    rw_mol = Chem.RWMol(new_mol)
    # Collect indexes of dummy atoms (atomic num 0)
    dummy_idxs = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    # print(f"[DEBUG] Found dummy atoms at indexes: {dummy_idxs}")

    # Remove dummy atoms in reverse order
    for idx in sorted(dummy_idxs, reverse=True):
        try:
            rw_mol.RemoveAtom(idx)
        except Exception as e:
            print(f"[ERROR] Failed to remove dummy atom at index {idx}: {e}")

    mol_clean = rw_mol.GetMol()

    # Restore conformer positions for the remaining atoms
    if positions is not None:
        new_conf = Chem.Conformer(mol_clean.GetNumAtoms())
        j = 0
        for i in range(len(positions)):
            # Only restore if this atom was not removed
            if i not in dummy_idxs:
                new_conf.SetAtomPosition(j, positions[i])
                j += 1
        mol_clean.RemoveAllConformers()
        mol_clean.AddConformer(new_conf, assignId=True)

    # Sanitize the molecule (without performing kekulization)
    try:
        Chem.SanitizeMol(mol_clean, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception as e:
        print("[ERROR] Sanitization error:", e)

    # print(f"[DEBUG] Final atom count after sanitization: {mol_clean.GetNumAtoms()}")
    return mol_clean


def replace_dummies_with_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Returns a copy of the molecule in which all dummy atoms (atomic number 0)
    are replaced with hydrogens (atomic number 1). The 3D conformer is preserved.
    This avoids sanitization/kekulization issues related to truly removing dummies.
    """
    # Deep copy the molecule
    new_mol = copy.deepcopy(mol)

    # Save conformer positions if available
    has_conf = new_mol.GetNumConformers() > 0
    if has_conf:
        conf = new_mol.GetConformer()
        positions = [conf.GetAtomPosition(i) for i in range(new_mol.GetNumAtoms())]
    else:
        positions = None

    # Iterate over atoms and if dummy, simply replace with hydrogen
    for atom in new_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)

    # Restore conformer positions
    if positions is not None:
        new_conf = Chem.Conformer(new_mol.GetNumAtoms())
        for i in range(new_mol.GetNumAtoms()):
            new_conf.SetAtomPosition(i, positions[i])
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(new_conf, assignId=True)

    # Try to sanitize without kekulization so that RDKit doesn't choke on aromaticity issues
    try:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception as e:
        print("[ERROR] Sanitization error:", e)

    return new_mol


def _apply_noise_to_fragment(mol: Chem.Mol, noise: float) -> Chem.Mol:
    """
    Apply Gaussian noise to the coordinates of a fragment molecule.

    Args:
        mol: The RDKit molecule to apply noise to.
        noise: Standard deviation of the Gaussian noise to add to each coordinate.

    Returns:
        A new RDKit molecule with perturbed coordinates.
    """
    conf = mol.GetConformer()
    noise_vector = np.random.normal(0, noise, 3)
    for i in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_pos = pos + noise_vector
        conf.SetAtomPosition(i, new_pos)
    return mol


def _explode_fragment_away_from_com(mol: Chem.Mol, reference_com: np.ndarray, offset_magnitude: float) -> Chem.Mol:
    """
    Displaces all coordinates in the fragment by a translation vector.
    The translation vector is computed as the normalized vector from the reference_com
    to the fragment's own CoM, scaled by offset_magnitude.

    Args:
        mol: RDKit Mol object with a 3D conformer.
        reference_com: (3,) NumPy array representing the reference center-of-mass.
        offset_magnitude: Magnitude (in Å) to translate the fragment.

    Returns:
        A new RDKit Mol with its coordinates displaced.
    """
    new_mol = Chem.Mol(mol)
    conf = new_mol.GetConformer(0)
    num_atoms = new_mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    # Compute fragment CoM (simple average over heavy atoms)
    frag_com = coords.mean(axis=0)
    direction = frag_com - reference_com
    norm = np.linalg.norm(direction)
    if norm == 0:
        # If fragment CoM coincides with reference, choose a random direction
        direction = np.random.randn(3)
        norm = np.linalg.norm(direction)
    translation = (direction / norm) * offset_magnitude
    new_coords = coords + translation
    for i, pos in enumerate(new_coords):
        conf.SetAtomPosition(i, pos)
    return new_mol


def _noise_fragment(mol: Chem.Mol, noise: float) -> Chem.Mol:
    """
    Apply Gaussian noise to the fragment coordinates

    Args:
        mol: The RDKit molecule to apply noise to.
        noise: Standard deviation of the Gaussian noise to add to each coordinate.

    Returns:
        A new RDKit molecule with perturbed coordinates.
    """
    conf = mol.GetConformer()
    noise_vector = np.random.normal(0, noise, 3)
    for i in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_pos = pos + noise_vector
        conf.SetAtomPosition(i, new_pos)
    return mol


def view_pocket_fragments_deprecated(
    pocket_mol: Chem.Mol,
    fragments: list[Chem.Mol],
    reference_com: np.ndarray | None = None,
    explode_offset: float = 0.0,
    noisy_offset: float = 0.0,
    pocket_style: str = "cartoon",
    fragment_style: str = "stick",
    fragment_colors: list[str] | None = None,
    width: int = 800,
    height: int = 500,
    marker: tuple[np.ndarray, dict[str, str]] | None = None,
) -> py3Dmol.view:
    """
    Visualize a pocket with fragments, handling small fragments correctly.
    """
    view = py3Dmol.view(width=width, height=height)

    # Maintain our own model counter
    model_counter = 0

    # Add pocket as model 0
    view.addModel(Chem.MolToPDBBlock(pocket_mol), "pdb")
    view.setStyle({"model": model_counter}, {pocket_style: {"color": "lightgrey"}})
    model_counter += 1

    # Prepare colors
    default_colors = ["red", "blue", "green", "orange", "purple", "magenta", "cyan"]
    fragment_colors = fragment_colors or default_colors
    color_cycle = islice(cycle(fragment_colors), len(fragments))

    for frag, color in zip(fragments, color_cycle):
        frag = replace_dummies_with_hydrogens(frag)

        # Handle fragment explosion if needed
        if reference_com is not None and explode_offset:
            frag = _explode_fragment_away_from_com(frag, reference_com, explode_offset)

        if noisy_offset:
            frag = _noise_fragment(frag, noisy_offset)
        # Small fragment handling
        num_heavy_atoms = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() > 1)
        if num_heavy_atoms < 1:
            # Add an empty model for the sphere
            view.addModel("", "xyz")
            pos = frag.GetConformer().GetAtomPosition(0)
            view.addSphere(
                {"center": {"x": pos.x, "y": pos.y, "z": pos.z}, "radius": 0.5, "color": color, "opacity": 1.0}
            )
        else:
            # Regular molecule fragment
            # pdb_block = Chem.MolToPDBBlock(frag)
            # print(pdb_block)
            # view.addModel(Chem.MolToPDBBlock(frag), "pdb")
            view.addModel(Chem.MolToMolBlock(frag), "mol")
        # Apply style to the model just added using our counter
        view.setStyle({"model": model_counter}, {fragment_style: {"color": color}})
        model_counter += 1

    # Add marker if specified
    if marker:
        center, options = marker
        view.addModel("", "xyz")
        view.addSphere(
            {
                "center": dict(zip(("x", "y", "z"), center)),
                "radius": options.get("radius", 1.5),
                "color": options.get("color", "magenta"),
                "opacity": options.get("opacity", 0.5),
            }
        )

    view.zoomTo()
    view.render()
    return view


def _view_pocket_fragments(  # noqa: C901
    pocket_mol: Chem.Mol,
    fragments: list[Chem.Mol],
    reference_ligand: Chem.Mol | None = None,
    reference_style: str = "stick",
    reference_color: str = "lightgreen",
    reference_com: np.ndarray | None = None,
    explode_offset: float = 0.0,
    # NO positional noisy_offset by default as requested
    noisy_offset: float = 0.0,
    pocket_style: str = "cartoon",
    fragment_style: str = "stick",
    fragment_colors: list[str] | None = None,
    width: int = 800,
    height: int = 500,
    marker: tuple[np.ndarray, dict[str, str]] | None = None,
    # New options for this request:
    fragment_opacity: float = 0.5,  # translucent base layer opacity
    max_rot_deg: float = 15.0,  # max random rotation applied to noisy overlay
    view_dummies: bool = False,  # whether to keep dummy atoms in fragments
    color_noise_level: float = 0.08,  # RGB perturbation amplitude for noisy overlay (0 = no change)
) -> py3Dmol.view:
    """
    Visualize a pocket with fragments. For each fragment:
      - add a translucent base copy with no positional noise
      - add a second copy rotated randomly around its COM and with a slightly noisy color (opaque)
    Small fragments (single atom) are rendered as spheres for both layers.

    Parameters relevant to the request:
      fragment_opacity: opacity for the base (clean) layer
      max_rot_deg: maximum rotation (degrees) to randomly apply to the noisy overlay
      color_noise_level: how much to perturb RGB channels for the noisy overlay (fractional)
    """
    view = py3Dmol.view(width=width, height=height)
    model_counter = 0

    # pocket base model
    view.addModel(Chem.MolToPDBBlock(pocket_mol), "pdb")
    view.setStyle({"model": model_counter}, {pocket_style: {"color": "lightgrey"}})
    model_counter += 1

    # Optionally add reference ligand (unchanged behavior)
    if reference_ligand is not None:
        if not view_dummies:
            ref = replace_dummies_with_hydrogens(reference_ligand)
        if reference_com is not None and explode_offset:
            ref = _explode_fragment_away_from_com(ref, reference_com, explode_offset)
        # draw reference (single, solid layer)
        num_heavy_atoms_ref = sum(1 for atom in ref.GetAtoms() if atom.GetAtomicNum() > 1)
        if num_heavy_atoms_ref < 1:
            view.addModel("", "xyz")
            pos = ref.GetConformer().GetAtomPosition(0)
            view.addSphere(
                {
                    "center": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "radius": 0.5,
                    "color": reference_color,
                    "opacity": 1.0,
                }
            )
        else:
            view.addModel(Chem.MolToMolBlock(ref), "mol")
        view.setStyle({"model": model_counter}, {reference_style: {"color": reference_color}})
        model_counter += 1

    default_colors = ["red", "blue", "green", "orange", "purple", "magenta", "cyan"]
    fragment_colors = fragment_colors or default_colors
    color_cycle = islice(cycle(fragment_colors), len(fragments))

    def perturb_color(color: str, noise_level: float) -> str:
        """Perturb an input color name/hex by small random RGB noise and return hex."""
        try:
            rgb = np.array(mcolors.to_rgb(color))
        except Exception:
            # fallback: if unknown color string, treat as gray
            rgb = np.array([0.5, 0.5, 0.5])
        # additive noise in [-noise_level, +noise_level]
        noise = (np.random.rand(3) * 2.0 - 1.0) * noise_level
        rgb_noisy = np.clip(rgb + noise, 0.0, 1.0)
        return mcolors.to_hex(rgb_noisy)

    def random_rotation_matrix(max_deg: float) -> np.ndarray:
        """Random small rotation matrix using axis-angle with angle in [-max_deg, max_deg]."""
        angle = math.radians(random.uniform(-max_deg, max_deg))
        axis = np.random.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ux, uy, uz = axis
        c = math.cos(angle)
        s = math.sin(angle)
        R = np.array(
            [
                [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
                [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
            ],
            dtype=float,
        )
        return R

    def apply_rotation_to_mol(mol: Chem.Mol, R: np.ndarray, center: np.ndarray) -> Chem.Mol:
        """Return a copy of mol whose conformer coordinates were rotated around center by R."""
        mol_copy = Chem.Mol(mol)  # shallow copy of molecule object/structure
        conf = mol_copy.GetConformer()
        n = mol_copy.GetNumAtoms()
        for i in range(n):
            pos = conf.GetAtomPosition(i)
            vec = np.array([pos.x, pos.y, pos.z]) - center
            vec_rot = R @ vec
            new = center + vec_rot
            conf.SetAtomPosition(i, Point3D(float(new[0]), float(new[1]), float(new[2])))
        return mol_copy

    for frag, base_color in zip(fragments, color_cycle):
        if not view_dummies:
            frag = replace_dummies_with_hydrogens(frag)

        # Optionally explode away from reference com
        if reference_com is not None and explode_offset:
            frag_exp = _explode_fragment_away_from_com(frag, reference_com, explode_offset)
        else:
            frag_exp = Chem.Mol(frag)  # copy

        # ---- Base (clean) translucent layer: no noise, no rotation ----
        num_heavy_atoms = sum(1 for atom in frag_exp.GetAtoms() if atom.GetAtomicNum() > 1)
        if num_heavy_atoms < 1:
            # single-atom / tiny fragment (draw sphere)
            view.addModel("", "xyz")
            pos = frag_exp.GetConformer().GetAtomPosition(0)
            view.addSphere(
                {
                    "center": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "radius": 0.5,
                    "color": base_color,
                    "opacity": fragment_opacity,
                }
            )
            # style applied to model (even though it's an xyz model)
            view.setStyle({"model": model_counter}, {"sphere": {"color": base_color, "opacity": fragment_opacity}})
            model_counter += 1
        else:
            # regular fragment as mol block
            view.addModel(Chem.MolToMolBlock(frag_exp), "mol")
            view.setStyle(
                {"model": model_counter}, {fragment_style: {"color": base_color, "opacity": fragment_opacity}}
            )
            model_counter += 1

        # ---- Noisy overlay: random rotation + color-noise, opaque ----
        # compute COM for rotation pivot
        conf = frag.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(frag.GetNumAtoms())])
        com = coords.mean(axis=0)  # center-of-mass approx by arithmetic mean

        R = random_rotation_matrix(max_rot_deg)

        # rotated copy
        frag_rot = apply_rotation_to_mol(frag, R, com)

        # optionally explode rotated copy as well
        if reference_com is not None and explode_offset:
            frag_rot = _explode_fragment_away_from_com(frag_rot, reference_com, explode_offset)

        noisy_color = perturb_color(base_color, color_noise_level)

        num_heavy_atoms_rot = sum(1 for atom in frag_rot.GetAtoms() if atom.GetAtomicNum() > 1)
        if num_heavy_atoms_rot < 1:
            view.addModel("", "xyz")
            pos = frag_rot.GetConformer().GetAtomPosition(0)
            view.addSphere(
                {
                    "center": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "radius": 0.5,
                    "color": noisy_color,
                    "opacity": 1.0,
                }
            )
            view.setStyle({"model": model_counter}, {"sphere": {"color": noisy_color, "opacity": 1.0}})
            model_counter += 1
        else:
            view.addModel(Chem.MolToMolBlock(frag_rot), "mol")
            view.setStyle({"model": model_counter}, {fragment_style: {"color": noisy_color, "opacity": 1.0}})
            model_counter += 1

    # Add marker if specified
    if marker:
        center, options = marker
        view.addModel("", "xyz")
        view.addSphere(
            {
                "center": dict(zip(("x", "y", "z"), center)),
                "radius": options.get("radius", 1.5),
                "color": options.get("color", "magenta"),
                "opacity": options.get("opacity", 0.5),
            }
        )
        model_counter += 1

    view.zoomTo()
    view.render()
    return view


def view_pocket_fragments(
    pocket_mol: Chem.Mol,
    fragments: list[Chem.Mol] | list[list[Chem.Mol]],
    reference_ligand: Chem.Mol | None = None,
    reference_style: str = "stick",
    reference_color: str = "lightgreen",
    pocket_style: str = "cartoon",
    pocket_alpha: float = 1.0,
    pocket_surface: bool = True,
    pocket_surface_opacity: float = 0.25,
    pocket_surface_color: str = "lightgrey",
    pocket_overlay: bool = True,
    fragment_style: str = "stick",
    fragment_colors: list[str] | None = None,
    width: int = 800,
    height: int = 500,
    marker: tuple[tuple[float, float, float] | np.ndarray, dict] | None = None,
    fragment_opacity: float = 0.5,
    ref_ligand_opacity: float = 1.0,
    view_dummies: bool = False,
):
    """
    Visualize a pocket with fragments. Supports single frame or trajectory (list of frames).

    - fragments: either list[Chem.Mol] (single frame) or list[list[Chem.Mol]] (trajectory).
    - Single frame: returns py3Dmol.view.
    - Trajectory: displays a slider + view and returns (output_widget, slider). Requires ipywidgets.

    Optional: pocket_surface, pocket_overlay, ref_ligand_opacity.
    """
    fragments = list(fragments)
    is_trajectory = len(fragments) > 0 and isinstance(fragments[0], (list, tuple))
    frames = fragments if is_trajectory else [fragments]

    default_colors = ["red", "blue", "green", "orange", "purple", "magenta", "cyan"]
    fragment_colors = fragment_colors or default_colors

    def _build_view(frame: list[Chem.Mol], prev_state=None):
        v = py3Dmol.view(width=width, height=height)
        m = 0

        v.addModel(Chem.MolToPDBBlock(pocket_mol), "pdb")
        try:
            v.setStyle(
                {"model": 0},
                {pocket_style: {"color": pocket_surface_color, "opacity": float(pocket_alpha)}},
            )
        except Exception:
            try:
                v.setStyle({"model": 0}, {pocket_style: {"color": pocket_surface_color}})
            except Exception:
                pass
        m += 1

        if pocket_surface:
            try:
                v.addSurface(
                    py3Dmol.VDW,
                    {"opacity": float(pocket_surface_opacity), "color": pocket_surface_color},
                    {"model": 0},
                )
            except Exception:
                try:
                    v.addSurface(
                        py3Dmol.VDW,
                        {"opacity": float(pocket_surface_opacity), "color": pocket_surface_color, "model": 0},
                    )
                except Exception:
                    pass

        if pocket_overlay:
            v.addModel(Chem.MolToPDBBlock(pocket_mol), "pdb")
            try:
                if pocket_style == "stick":
                    v.setStyle(
                        {"model": m},
                        {"stick": {"color": pocket_surface_color, "opacity": float(pocket_alpha)}},
                    )
                elif pocket_style == "cartoon":
                    v.setStyle(
                        {"model": m},
                        {"cartoon": {"color": pocket_surface_color, "opacity": float(pocket_alpha)}},
                    )
                else:
                    v.setStyle(
                        {"model": m},
                        {pocket_style: {"color": pocket_surface_color, "opacity": float(pocket_alpha)}},
                    )
            except Exception:
                try:
                    v.setStyle({"model": m}, {pocket_style: {"color": pocket_surface_color}})
                except Exception:
                    pass
            m += 1

        if reference_ligand is not None:
            ref = (
                replace_dummies_with_hydrogens(reference_ligand)
                if not view_dummies
                else reference_ligand
            )
            n_heavy_ref = sum(1 for a in ref.GetAtoms() if a.GetAtomicNum() > 1)
            if n_heavy_ref < 1:
                v.addModel("", "xyz")
                pos = ref.GetConformer().GetAtomPosition(0)
                v.addSphere(
                    {
                        "center": {"x": pos.x, "y": pos.y, "z": pos.z},
                        "radius": 0.5,
                        "color": reference_color,
                        "opacity": ref_ligand_opacity,
                    }
                )
                v.setStyle(
                    {"model": m},
                    {"sphere": {"color": reference_color, "opacity": ref_ligand_opacity}},
                )
                m += 1
            else:
                v.addModel(Chem.MolToMolBlock(ref), "mol")
                v.setStyle(
                    {"model": m},
                    {reference_style: {"color": reference_color, "opacity": ref_ligand_opacity}},
                )
                m += 1

        cc = islice(cycle(fragment_colors), len(frame))
        for frag, ccol in zip(frame, cc):
            if frag is None:
                continue
            if not view_dummies:
                frag = replace_dummies_with_hydrogens(frag)
            frag_base = Chem.Mol(frag)
            n_heavy = sum(1 for a in frag_base.GetAtoms() if a.GetAtomicNum() > 1)
            if n_heavy < 1:
                v.addModel("", "xyz")
                pos = frag_base.GetConformer().GetAtomPosition(0)
                v.addSphere(
                    {
                        "center": {"x": pos.x, "y": pos.y, "z": pos.z},
                        "radius": 0.5,
                        "color": ccol,
                        "opacity": float(fragment_opacity),
                    }
                )
                v.setStyle(
                    {"model": m},
                    {"sphere": {"color": ccol, "opacity": float(fragment_opacity)}},
                )
                m += 1
            else:
                v.addModel(Chem.MolToMolBlock(frag_base), "mol")
                v.setStyle(
                    {"model": m},
                    {fragment_style: {"color": ccol, "opacity": float(fragment_opacity)}},
                )
                m += 1
                frag_overlay = Chem.Mol(frag)
                v.addModel(Chem.MolToMolBlock(frag_overlay), "mol")
                v.setStyle({"model": m}, {fragment_style: {"color": ccol, "opacity": 1.0}})
                m += 1

        if marker:
            center, opts = marker
            center = tuple(center) if hasattr(center, "__iter__") and not isinstance(center, dict) else center
            v.addModel("", "xyz")
            v.addSphere(
                {
                    "center": dict(zip(("x", "y", "z"), center)),
                    "radius": opts.get("radius", 1.5),
                    "color": opts.get("color", "magenta"),
                    "opacity": opts.get("opacity", 0.5),
                }
            )
            m += 1

        v.zoomTo()
        if prev_state is not None:
            for setter in ("setState", "setViewerState", "setCamera"):
                try:
                    getattr(v, setter)(prev_state)
                    break
                except Exception:
                    pass
        v.render()
        return v

    if not is_trajectory:
        return _build_view(frames[0], prev_state=None)

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as e:
        raise ImportError(
            "Trajectory view requires ipywidgets. Install with: pip install ipywidgets"
        ) from e

    out = widgets.Output()
    slider = widgets.IntSlider(
        value=len(frames) - 1,
        min=0,
        max=len(frames) - 1,
        step=1,
        description="frame",
    )
    last_viewer_state = None
    last_view_obj = None

    def _capture_view_state(view_obj):
        for getter in ("getState", "getViewerState", "getCamera"):
            try:
                fn = getattr(view_obj, getter, None)
                if fn is not None and (st := fn()) is not None:
                    return st
            except Exception:
                continue
        return None

    def _render(i):
        nonlocal last_viewer_state, last_view_obj
        if last_view_obj is not None:
            st = _capture_view_state(last_view_obj)
            if st is not None:
                last_viewer_state = st
        with out:
            out.clear_output(wait=True)
            v = _build_view(frames[i], prev_state=last_viewer_state)
            display(v)
            last_view_obj = v

    _render(len(frames) - 1)
    slider.observe(lambda change: _render(int(change["new"])) if change["name"] == "value" else None, names="value")
    display(widgets.VBox([slider, out]))
    return out, slider
