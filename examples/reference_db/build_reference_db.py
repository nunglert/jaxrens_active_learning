"""Example: build a DFT-labelled reference database for the dashboard.

Two dashboard panels want a *reference* set of configs the AL loop does not
itself produce:

  * the **EOS** panel (``jaxrens_al.panels.eos``) needs, per crystalline phase,
    a volume scan with DFT energies -- E(V) / H(P) it fits Birch-Murnaghan to;
  * the **atomic-environments PCA** (``atomic_envs``) is far more informative
    when *fitted on a proper reference database* (spanning phases + volumes +
    rattlings) than on whatever the sampler happened to visit.

This script builds both from the Materials Project and labels them with VASP at
**the same DFT level as the AL loop** -- by reusing the very same
``VASPEvaluator`` configuration (edit ``DFT_INCAR`` / the cluster block to match
YOUR ``run_al.py``; that is what makes the levels identical). The recipe:

  1. fetch the element's experimentally observed phases (``fetch_mp_elemental``);
  2. per phase, a zero-rattle volume scan  -> ``eos_<phase>.extxyz`` (EOS);
  3. a rattled+scaled pool across all phases -> extra diversity for the PCA;
  4. DFT-label every set with one shared evaluator;
  5. concatenate the labelled EOS scans + pool into ``reference_db.extxyz``.

Like the AL examples this is a CONCRETE TEMPLATE -- the ``DFT_INCAR`` below is a
real-shaped placeholder; edit it to match YOUR ``run_al.py``. The DFT cluster is
resolved by name from a CLI argument (``--cluster <name>`` -> ``<name>.json``
under ``REMOTE_HANDLER_CONFIG_DIR``, the same registry your AL loop uses)::

    python build_reference_db.py --cluster dirona --element Sn

Needs the ``[run]`` + ``[refdb]`` extras, a configured cluster
(``REMOTE_HANDLER_CONFIG_DIR`` + SSH config) and ``MP_API_KEY``. Set
``STUB = True`` first to exercise the wiring without a cluster or VASP (the
cluster is still resolved, and the Materials-Project fetch still runs, so a
config + key/network are needed even then).

Render the outputs afterwards through the existing dashboard wiring:

  * EOS curves on the AL page -- each iteration's model is scored on the
    per-phase scans (needs a trained model in the run)::

        jaxrens-al-dashboard <calchome> --eos-reference reference_db/eos_*.extxyz

    (equivalently an ``eos-reference: [...]`` key in the calchome's
    ``dashboard.yaml``).

  * PCA of atomic environments + energy/volume spread of the reference
    database, via the standalone DFT section::

        jaxrens-al-dashboard dft reference_db/reference_db.extxyz
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows import evaluators

from jaxrens_al.reference_db import (
    eos_phase_scans,
    fetch_mp_elemental,
    rattle_atoms,
)
from jaxrens_al.reference_eval import concatenate_configs, evaluate_config_sets

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- toggles --------------------------------------------------------------
STUB = False  # True: mock DFT locally (no cluster/VASP); MP fetch still runs.
ELEMENT = "Sn"

# --- reference-database grid ---------------------------------------------
TARGET_CUTOFF = 4.0  # supercell must host this (Å); match the model's r_cut.
# EOS: a fine volume scan at zero rattle (>=5 points for a stable BM fit).
EOS_VOL_SCALINGS = np.linspace(0.90, 1.10, 21)
# PCA pool: coarser volumes, non-zero rattle for local-environment diversity.
POOL_VOL_SCALINGS = np.linspace(0.94, 1.06, 5)
POOL_RATTLE_STRENGTHS = (0.1, 0.2)
POOL_REALIZATIONS = 2

# --- output ---------------------------------------------------------------
HOME = Path(os.getcwd())
OUT_DIR = HOME / "reference_db"
WORK_DIR = HOME / "reference_db_work"

# --- DFT level (KEEP IN SYNC WITH YOUR run_al.py VASPEvaluator) -----------
DFT_INCAR = {
    "xc": "R2SCAN",
    "encut": 300,
    "prec": "Normal",
    "ismear": -1,
    "sigma": 0.025,
    "ibrion": -1,
    "ediff": 1e-5,
    "kspacing": 0.3,
    "lwave": False,
    "lcharg": False,
    "lreal": False,
}


def make_evaluator(dft_cluster) -> "evaluators.VASPEvaluator":
    """The AL loop's VASP evaluator, reused verbatim so the level matches.

    The file system follows from the cluster: VASP runs on
    ``dft_cluster.remote_machine``, so evaluation stages through a
    ``RemoteFileSystem`` bound to that same alias (mirrors
    ``processing_resolver.execution_context``).
    """
    return evaluators.VASPEvaluator(
        remote_fs=RemoteFileSystem(dft_cluster.remote_machine),
        cluster_config=dft_cluster,
        num_parallel=25,
        incar_kwargs=DFT_INCAR,
        stub=STUB,
        loop_time=10,
        remote_savepath=dft_cluster.savepath,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a DFT-labelled reference database (EOS + PCA) for "
        "the dashboard."
    )
    parser.add_argument(
        "--cluster",
        required=True,
        help="Name of the DFT ClusterConfig to run VASP on -- loads "
        "<name>.json from REMOTE_HANDLER_CONFIG_DIR (as configured for your "
        "AL loop). Loaded even in STUB mode (the stub evaluator ignores it).",
    )
    parser.add_argument(
        "--element",
        default=ELEMENT,
        help=f"Chemical symbol to build the database for (default: {ELEMENT}).",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("REMOTE_HANDLER_CONFIG_DIR", str(HOME / "config"))
    dft_cluster = ClusterConfig.from_name(args.cluster)

    # 1. Fetch the element's experimentally observed phases.
    phases = fetch_mp_elemental(args.element)
    logging.info(
        "Fetched %d %s phase(s) from Materials Project.",
        len(phases),
        args.element,
    )

    # 2. Per-phase zero-rattle volume scans (one EOS curve each).
    eos_scans = eos_phase_scans(
        phases, vol_scalings=EOS_VOL_SCALINGS, target_cutoff=TARGET_CUTOFF
    )

    # 3. A rattled + volume-scaled pool across all phases (PCA diversity).
    pool = rattle_atoms(
        phases,
        rattle_strengths=POOL_RATTLE_STRENGTHS,
        vol_scalings=POOL_VOL_SCALINGS,
        num_realizations=POOL_REALIZATIONS,
        target_cutoff=TARGET_CUTOFF,
        seed=0,
    )

    # 4. Label every set with one shared evaluator (identical DFT level).
    sets = {f"eos_{label}": scan for label, scan in eos_scans.items()}
    sets["rattled_pool"] = pool
    labelled = evaluate_config_sets(
        make_evaluator(dft_cluster), sets, WORK_DIR
    )

    # 5. Publish: per-phase EOS files + one concatenated reference database.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import ase.io

    eos_files = []
    for label, path in labelled.items():
        if label == "rattled_pool":
            continue
        dest = OUT_DIR / f"{label}.extxyz"
        ase.io.write(str(dest), ase.io.read(str(path), index=":"))
        eos_files.append(dest)

    ref_db = OUT_DIR / "reference_db.extxyz"
    n = concatenate_configs(list(labelled.values()), ref_db)

    logging.info("Wrote %d EOS phase file(s) to %s", len(eos_files), OUT_DIR)
    logging.info("Wrote %d-config reference database -> %s", n, ref_db)
    eos_args = " ".join(str(f) for f in eos_files)
    print("\nRender these through the dashboard:")
    print("  # EOS curves on the AL page (scores the run's model per phase):")
    print(f"  jaxrens-al-dashboard <calchome> --eos-reference {eos_args}")
    print("  # PCA + energy/volume of the reference database (DFT section):")
    print(f"  jaxrens-al-dashboard dft {ref_db}")


if __name__ == "__main__":
    main()
