"""ProtFlow runner for SigmaDock molecular docking.

SigmaDock runs diffusion-based docking over a data directory organised as one
subfolder per protein-ligand complex. This runner:

1. Splits each input complex PDB into a protein-only PDB (ATOM records) and a
   ligand SDF (HETATM records parsed via RDKit), then builds the expected
   SigmaDock data directory layout.
2. Writes a temporary experiment config YAML (deleted after the run).
3. Submits one ``python scripts/sample.py`` call, batching all poses so the
   model checkpoint is loaded only once.
4. Parses ``predictions.pt`` / ``rescoring.pt`` / ``posebusters.pt`` into the
   canonical ProtFlow scores dataframe.
5. Exports each docked pose as a protein-ligand complex PDB (protein ATOM
   records + docked ligand as HETATM). ``location`` therefore stays a PDB
   throughout the ProtFlow pipeline. The ligand-only SDF is also written and
   stored in a ``<prefix>_docked_sdf`` column.

Input poses are **complex PDB files** that contain both protein (ATOM) and
ligand (HETATM) records — the standard ProtFlow pose format.

Config keys (protflow config):
    SIGMADOCK_SCRIPT_PATH   absolute path to scripts/sample.py
    SIGMADOCK_PYTHON_PATH   python executable inside the sigmadock conda env
    SIGMADOCK_CKPT_DIR      path to the checkpoint directory (contains .ckpt)
    SIGMADOCK_PRE_CMD       optional prefix command, e.g. ``conda activate ...``
"""

from __future__ import annotations

import logging
import os
import shutil
from glob import glob
from pathlib import Path

import pandas as pd

from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput, prepend_cmd


# ---------------------------------------------------------------------------
# Runner class
# ---------------------------------------------------------------------------

class SigmaDockRunner(Runner):
    """ProtFlow runner for SigmaDock diffusion-based molecular docking.

    Input poses are complex PDBs (protein ATOM + ligand HETATM). The runner
    splits each complex, runs SigmaDock on the full batch (one model load),
    and returns complex PDBs with the docked ligand coordinates.

    Parameters
    ----------
    script_path:
        Absolute path to ``scripts/sample.py`` inside the sigmadock project.
    python_path:
        Python executable (or ``python`` if already on PATH) for the env that
        has sigmadock installed.
    ckpt_dir:
        Path to the directory that holds the ``.ckpt`` checkpoint file.
    pre_cmd:
        Optional shell prefix inserted before the python call.
    jobstarter:
        Default jobstarter for this runner instance.
    """

    def __init__(
        self,
        script_path: str | None = None,
        python_path: str | None = None,
        ckpt_dir: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        config = require_config()

        self.script_path = script_path or load_config_path(config, "SIGMADOCK_SCRIPT_PATH")
        self.python_path = python_path or load_config_path(config, "SIGMADOCK_PYTHON_PATH")
        self.ckpt_dir = ckpt_dir or load_config_path(config, "SIGMADOCK_CKPT_DIR")
        self.pre_cmd = pre_cmd or load_config_path(config, "SIGMADOCK_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "sigmadock_runner"
        self.index_layers = 0

        # scripts/sample.py is one level below the project root.
        self._sigmadock_root = Path(self.script_path).resolve().parent.parent

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------
    # Public run entry point
    # ------------------------------------------------------------------

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        options: str | None = None,
        pose_options: list[str] | str | None = None,  # noqa: ARG002  (reserved for future use)
        include_scores: list[str] | None = None,
        overwrite: bool = False,
        run_tag: str = "protflow_run",
        seed: int = 0,
        num_seeds: int = 1,
        ligand_chain: str | None = None,
    ) -> Poses:
        """Run SigmaDock on all input poses as a single batched experiment.

        Parameters
        ----------
        poses:
            Input complex PDB poses (protein ATOM + ligand HETATM records).
        prefix:
            ProtFlow run prefix used for the work directory and score file.
        jobstarter:
            Overrides the instance-level jobstarter for this call only.
        options:
            Extra Hydra overrides appended verbatim to the SigmaDock command.
        include_scores:
            Opt-in to heavy score fields. Currently supports
            ``"posebusters_checks"`` for the full per-check dict.
        overwrite:
            Re-run even if a score file from a previous run exists.
        run_tag:
            Passed to ``sampling.run_tag``; part of the output path.
        seed:
            Random seed for the diffusion sampler.
        num_seeds:
            Number of independent sampling seeds per complex.
        ligand_chain:
            If provided, only HETATM records from this chain ID are treated as
            the ligand when splitting the complex PDB. Useful when the PDB
            contains solvent or other small molecules on separate chains.
        """
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))

        # ------------------------------------------------------------------
        # Reuse cached score file when available.
        # ------------------------------------------------------------------
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (
            scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)
        ) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)
            return RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # ------------------------------------------------------------------
        # Split complex PDBs and build the SigmaDock data directory layout.
        # ------------------------------------------------------------------
        data_dir = os.path.join(work_dir, "data")
        experiment_name = prefix.replace("/", "_").replace(" ", "_")
        experiment_dir = os.path.join(data_dir, experiment_name)
        self._setup_data_dir(
            poses=poses,
            experiment_dir=experiment_dir,
            ligand_chain=ligand_chain,
        )

        # ------------------------------------------------------------------
        # Write the temporary experiment YAML, run SigmaDock, clean up.
        # ------------------------------------------------------------------
        output_dir = os.path.join(work_dir, "sampling_output")
        temp_yaml = self._write_experiment_yaml(name=experiment_name)

        try:
            cmd = self.write_cmd(
                experiment_name=experiment_name,
                data_dir=data_dir,
                output_dir=output_dir,
                run_tag=run_tag,
                seed=seed,
                num_seeds=num_seeds,
                extra_args=options or "",
            )

            if self.pre_cmd:
                cmd = prepend_cmd(cmds=[cmd], pre_cmd=self.pre_cmd)[0]

            jobstarter.start(
                cmds=[cmd],
                jobname=self.name,
                wait=True,
                output_path=work_dir,
            )
        finally:
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
                logging.debug("Removed temporary experiment YAML: %s", temp_yaml)

        # ------------------------------------------------------------------
        # Parse outputs and build score dataframe.
        # ------------------------------------------------------------------
        predictions_path = self._find_predictions_pt(
            output_dir=output_dir,
            experiment_name=experiment_name,
            run_tag=run_tag,
            seed=seed,
        )

        complex_out_dir = os.path.join(work_dir, "complexes")
        sdf_out_dir = os.path.join(work_dir, "sdf_out")
        scores = collect_scores(
            predictions_path=predictions_path,
            complex_out_dir=complex_out_dir,
            sdf_out_dir=sdf_out_dir,
            include_scores=include_scores,
        )

        if len(scores.index) == 0:
            raise RuntimeError(
                f"{self}: collect_scores returned no rows. "
                f"Check {predictions_path} and runner logs in {work_dir}."
            )

        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def write_cmd(
        self,
        experiment_name: str,
        data_dir: str,
        output_dir: str,
        run_tag: str,
        seed: int,
        num_seeds: int,
        extra_args: str,
    ) -> str:
        """Return the SigmaDock sampling shell command."""
        return (
            f"cd '{self._sigmadock_root}' && "
            f"{self.python_path} {self.script_path} "
            f"sampling.experiments.name='{experiment_name}' "
            f"sampling.run_tag='{run_tag}' "
            f"sampling.graph.sample_conformer=true "
            f"sampling.graph.fragmentation_strategy=canonical "
            f"sampling.seed={seed} "
            f"sampling.num_seeds={num_seeds} "
            f"sampling.output_dir='{output_dir}' "
            f"sampling.data.data_dir='{data_dir}' "
            f"sampling.hardware.devices=auto "
            f"sampling.model.ckpt_dir='{self.ckpt_dir}' "
            f"hydra.run.dir='{output_dir}/hydra_out' "
            f"{extra_args}"
        ).strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_data_dir(
        self, poses: Poses, experiment_dir: str, ligand_chain: str | None
    ) -> None:
        """Split each complex PDB and populate the SigmaDock data directory.

        For each pose (complex PDB), produces::

            experiment_dir/<description>/
                protein.pdb   ← ATOM records only
                ligand.sdf    ← HETATM records parsed into SDF via RDKit

        Parameters
        ----------
        ligand_chain:
            Restrict HETATM extraction to this chain ID. When None, all HETATM
            records are treated as ligand (excluding water: resname HOH/WAT).
        """
        os.makedirs(experiment_dir, exist_ok=True)

        for _, row in poses.df.iterrows():
            complex_pdb: str = row["location"]
            desc: str = row["description"]

            complex_dir = os.path.join(experiment_dir, desc)
            os.makedirs(complex_dir, exist_ok=True)

            protein_dest = os.path.join(complex_dir, "protein.pdb")
            ligand_dest = os.path.join(complex_dir, "ligand.sdf")

            if not os.path.exists(protein_dest) or not os.path.exists(ligand_dest):
                _split_complex_pdb(
                    complex_pdb=complex_pdb,
                    protein_out=protein_dest,
                    ligand_out=ligand_dest,
                    ligand_chain=ligand_chain,
                )

    def _write_experiment_yaml(self, name: str) -> str:
        """Write a temporary experiment YAML to conf/experiments/ and return its path.

        Regexes are fixed to match the filenames written by ``_setup_data_dir``:
        ``protein.pdb`` and ``ligand.sdf``.
        """
        conf_dir = self._sigmadock_root / "conf" / "experiments"
        yaml_path = str(conf_dir / f"{name}.yaml")

        content = (
            f'_target_: sigmadock.experiments.ExperimentConfig\n'
            f'name: "{name}"\n'
            f'dataset: "{name}/"\n'
            f'pdb_regex: ".*\\.pdb$"\n'
            f'sdf_regex: ".*\\.sdf$"\n'
        )

        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        logging.debug("Wrote temporary experiment YAML: %s", yaml_path)
        return yaml_path

    def _find_predictions_pt(
        self, output_dir: str, experiment_name: str, run_tag: str, seed: int
    ) -> str:
        """Locate predictions.pt after a completed run.

        Expected pattern::

            <output_dir>/results/<experiment>/<checkpoint>/<run_tag>/seed_<N>/predictions.pt
        """
        pattern = os.path.join(
            output_dir, "results", experiment_name, "*", run_tag, f"seed_{seed}", "predictions.pt"
        )
        matches = sorted(glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No predictions.pt found. Searched: {pattern}")
        if len(matches) > 1:
            logging.warning("Multiple predictions.pt found; using most recent: %s", matches[-1])
        return matches[-1]

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        for subdir in ("data", "sampling_output", "complexes", "sdf_out"):
            target = os.path.join(work_dir, subdir)
            if os.path.isdir(target):
                shutil.rmtree(target)
                logging.debug("Removed previous output directory: %s", target)


# ---------------------------------------------------------------------------
# Complex PDB splitting
# ---------------------------------------------------------------------------

def _split_complex_pdb(
    complex_pdb: str,
    protein_out: str,
    ligand_out: str,
    ligand_chain: str | None,
) -> None:
    """Split a complex PDB into a protein PDB and a ligand SDF.

    Protein file: ATOM (and HETATM records that are water or not the ligand
    chain) written as plain PDB text — no RDKit needed.

    Ligand SDF: HETATM records (excluding HOH/WAT, filtered by chain if
    ``ligand_chain`` is set) loaded via RDKit and written as SDF so that
    SigmaDock gets correct bond orders and topology.

    Raises
    ------
    ValueError
        If no ligand HETATM records are found in the complex PDB.
    """
    from rdkit.Chem import MolFromPDBBlock, SDWriter

    protein_lines: list[str] = []
    hetatm_lines: list[str] = []
    water_resnames = {"HOH", "WAT", "SOL"}

    with open(complex_pdb, "r", encoding="utf-8") as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "ATOM":
                protein_lines.append(line)
            elif record == "HETATM":
                resname = line[17:20].strip()
                chain = line[21].strip()
                if resname in water_resnames:
                    # Exclude solvent from both protein and ligand.
                    continue
                if ligand_chain is not None and chain != ligand_chain:
                    # Non-target chain HETATMs go to the protein file
                    # (e.g. cofactors covalently bound to protein).
                    protein_lines.append(line)
                else:
                    hetatm_lines.append(line)
            elif record in ("TER", "END", "ENDMDL"):
                protein_lines.append(line)

    if not hetatm_lines:
        raise ValueError(
            f"No ligand HETATM records found in {complex_pdb}. "
            "Check that the file contains ligand coordinates or specify ligand_chain."
        )

    # Write protein PDB (no RDKit needed).
    with open(protein_out, "w", encoding="utf-8") as fh:
        fh.writelines(protein_lines)
        if not protein_lines or protein_lines[-1].strip() not in ("END", "ENDMDL"):
            fh.write("END\n")

    # Write ligand SDF via RDKit to preserve bond orders.
    hetatm_block = "".join(hetatm_lines) + "END\n"
    mol = MolFromPDBBlock(hetatm_block, sanitize=True, removeHs=False)
    if mol is None:
        # Fallback: write the raw HETATM lines as a minimal PDB-format SDF.
        # RDKit could not parse the HETATM block — the SDF will lack bond
        # orders but SigmaDock can still use it for coordinates.
        logging.warning(
            "RDKit could not parse ligand from %s; writing coordinate-only SDF.", complex_pdb
        )
        with open(ligand_out, "w", encoding="utf-8") as fh:
            fh.write(hetatm_block)
        return

    with SDWriter(ligand_out) as writer:
        writer.write(mol)


# ---------------------------------------------------------------------------
# Module-level score collection (callable without a runner instance)
# ---------------------------------------------------------------------------

def _write_sdf(mol, coords, out_path: str) -> None:
    """Update conformer coordinates of an RDKit Mol and write to SDF."""
    from rdkit.Chem import SDWriter
    from rdkit.Geometry import Point3D

    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    with SDWriter(out_path) as writer:
        writer.write(mol)


def _write_complex_pdb(protein_pdb: str, mol, coords, out_path: str) -> None:
    """Write a complex PDB: protein ATOM records + docked ligand as HETATM.

    Reads ATOM/TER lines from ``protein_pdb``, then appends ligand atoms
    formatted as HETATM records using the docked coordinates.

    Parameters
    ----------
    protein_pdb:
        Path to the protein-only PDB written by ``_split_complex_pdb``.
    mol:
        RDKit Mol object (``entry["lig_ref"]``) with correct atom ordering.
    coords:
        Tensor or array of shape ``(N_atoms, 3)`` — docked coordinates
        (``entry["x0_hat"]``).
    out_path:
        Destination complex PDB path.
    """
    protein_lines: list[str] = []
    with open(protein_pdb, "r", encoding="utf-8") as fh:
        for line in fh:
            record = line[:6].strip()
            if record in ("ATOM", "TER", "REMARK", "HEADER", "CRYST1"):
                protein_lines.append(line)

    # Build HETATM block from docked coordinates.
    hetatm_lines: list[str] = []
    resname = "LIG"
    chain = "Z"
    atom_serial = 9000  # high serial to avoid clashing with protein

    for i, atom in enumerate(mol.GetAtoms()):
        x, y, z = [float(v) for v in coords[i].tolist()]
        symbol = atom.GetSymbol()
        atom_name = f"{symbol}{i + 1}"[:4].ljust(4)
        line = (
            f"HETATM{atom_serial + i:5d} {atom_name} {resname} {chain}   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2}\n"
        )
        hetatm_lines.append(line)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(protein_lines)
        fh.writelines(hetatm_lines)
        fh.write("END\n")


def collect_scores(
    predictions_path: str,
    complex_out_dir: str,
    sdf_out_dir: str,
    include_scores: list[str] | None = None,
) -> pd.DataFrame:
    """Parse SigmaDock output files and return the canonical scores dataframe.

    For each complex × pose, exports:
    - A complex PDB (protein + docked ligand) to ``complex_out_dir``.
    - A ligand-only SDF to ``sdf_out_dir``.

    The protein PDB used for the complex comes from ``entry["pdb_path"]``
    (the original protein file written during data directory setup).

    Required output columns
    -----------------------
    ``description``
        Complex name. When ``num_seeds > 1``, each row gets
        ``<name>_pose_<i>`` so descriptions stay unique.
    ``location``
        Absolute path to the exported complex PDB (ProtFlow-native format).

    Additional score columns
    ------------------------
    ``docked_sdf``       Absolute path to the ligand-only SDF.
    ``vinardo_score``    Vinardo affinity from rescoring.pt (if present).
    ``rmsd``             RMSD vs. reference from posebusters.pt (if present).
    ``posebusters_pass`` True if all PoseBusters validity checks passed.
    ``seed``             Random seed used.

    Opt-in heavy fields (pass key in ``include_scores``)
    -----------------------------------------------------
    ``"posebusters_checks"``  Full per-check dict as a JSON string.
    """
    import torch

    include_set = set(include_scores or [])
    os.makedirs(complex_out_dir, exist_ok=True)
    os.makedirs(sdf_out_dir, exist_ok=True)

    run_dir = os.path.dirname(predictions_path)
    data = torch.load(predictions_path, map_location="cpu", weights_only=False)
    results: dict = data.get("results", {})

    rescoring_path = os.path.join(run_dir, "rescoring.pt")
    posebusters_path = os.path.join(run_dir, "posebusters.pt")

    rescoring: dict = (
        torch.load(rescoring_path, map_location="cpu", weights_only=False)
        if os.path.exists(rescoring_path) else {}
    )
    posebusters: dict = (
        torch.load(posebusters_path, map_location="cpu", weights_only=False)
        if os.path.exists(posebusters_path) else {}
    )

    rows: list[dict] = []

    for complex_name, entries in results.items():
        multi_seed = len(entries) > 1

        for i, entry in enumerate(entries):
            seed = entry.get("seed", i)
            mol = entry.get("lig_ref")
            coords = entry.get("x0_hat")
            protein_pdb: str = entry.get("pdb_path", "")

            description = f"{complex_name}_pose_{i}" if multi_seed else complex_name

            # Export ligand SDF.
            sdf_path = os.path.join(sdf_out_dir, f"{description}.sdf")
            if mol is not None and coords is not None:
                try:
                    _write_sdf(mol, coords, sdf_path)
                except Exception:
                    logging.warning("Could not write SDF for %s", description, exc_info=True)
                    sdf_path = None
            else:
                logging.warning("Missing lig_ref or x0_hat for %s — skipping", description)
                sdf_path = None

            # Export complex PDB (protein + docked ligand).
            complex_pdb_path = os.path.join(complex_out_dir, f"{description}_complex.pdb")
            if mol is not None and coords is not None and os.path.exists(protein_pdb):
                try:
                    _write_complex_pdb(protein_pdb, mol, coords, complex_pdb_path)
                except Exception:
                    logging.warning(
                        "Could not write complex PDB for %s", description, exc_info=True
                    )
                    complex_pdb_path = sdf_path or ""
            else:
                # Fall back to SDF if protein PDB is unavailable.
                complex_pdb_path = sdf_path or ""

            row: dict = {
                "description": description,
                "location": os.path.abspath(complex_pdb_path) if complex_pdb_path else "",
                "docked_sdf": os.path.abspath(sdf_path) if sdf_path else "",
                "seed": seed,
            }

            # Vinardo affinity score.
            if complex_name in rescoring:
                rs_entries = rescoring[complex_name]
                if i < len(rs_entries):
                    rs = rs_entries[i]
                    row["vinardo_score"] = rs.get("vinardo_score") if isinstance(rs, dict) else rs

            # PoseBusters validity and RMSD.
            if complex_name in posebusters:
                pb_entries = posebusters[complex_name]
                if i < len(pb_entries):
                    pb = pb_entries[i]
                    if isinstance(pb, dict):
                        row["rmsd"] = pb.get("rmsd")
                        bool_checks = {k: v for k, v in pb.items() if isinstance(v, bool)}
                        row["posebusters_pass"] = all(bool_checks.values()) if bool_checks else None
                        if "posebusters_checks" in include_set:
                            import json
                            row["posebusters_checks"] = json.dumps(bool_checks)

            rows.append(row)

    return pd.DataFrame(rows)
