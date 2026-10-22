"""Identify the phases of a nested-sampling (T, P) phase diagram.

Picks the most likely stored configuration per (T, P) grid cell from the NS
weights, relaxes it (positions at fixed cell, then cell under the replica's
pressure) and labels it with its spglib space group. A torch MACE ``.model``
is relaxed with torch-sim; a NeuralIL ``.pkl`` through its jaxrens backend,
with the NS run's ``supercell_trafo`` and neighbour ladder (read from
``--config``'s ``backend`` block unless given). See
:mod:`jaxrens_al.run.phase_grid`.

Example:
  python scripts/phase_grid.py results/output --config results/ns_config.yaml \\
      --model mace-matpes-r2scan-0.model --temperatures 150 300 500 700 1000 \\
      --workdir phase_grid --device cpu
  python scripts/phase_grid.py results/output --config results/ns_config.yaml \\
      --model trainer_it0_0.pkl --temperatures 300 800 --workdir phase_grid
"""

import argparse
import logging

from jaxrens_al.run.phase_grid import run_phase_grid


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dir", help="jaxrens NS output dir (<prefix>.runNN.* files)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="torch MLIP file (MACE .model) or NeuralIL .pkl",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="model backend (default: neuralil for a .pkl, else mace)",
    )
    parser.add_argument(
        "--supercell-trafo",
        type=int,
        nargs=3,
        default=None,
        help="jaxrens backend image expansion (default: from --config)",
    )
    parser.add_argument(
        "--max-neighbors-list",
        type=int,
        nargs="+",
        default=None,
        help="jaxrens neighbour-bucket ladder (default: from --config)",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        required=True,
        help="grid temperatures (K)",
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="where picks, jobs and results are written",
    )
    parser.add_argument(
        "--config", help="NS config yaml (per-replica pressures)"
    )
    parser.add_argument("--prefix", help="NS run prefix (default: inferred)")
    parser.add_argument(
        "--replicas",
        type=int,
        nargs="+",
        help="replica indices (default: all)",
    )
    parser.add_argument(
        "--cluster",
        default="local",
        help="'local' or a ClusterConfig JSON name/path",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["positions", "cell"],
        choices=["positions", "cell"],
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="spglib tolerance of the stored label",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="relax again even if results exist",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    backend = args.backend or (
        "neuralil" if args.model.endswith(".pkl") else "mace"
    )
    jax_kw = {}
    if not (backend == "mace" and args.model.endswith(".model")):
        block = {}
        if args.config:
            import yaml

            with open(args.config) as fh:
                block = (yaml.safe_load(fh) or {}).get("backend") or {}
        jax_kw = dict(
            supercell_trafo=args.supercell_trafo
            or block.get("supercell_trafo"),
            max_neighbors_list=args.max_neighbors_list
            or block.get("max_neighbors_list"),
            max_neighbors_offset=int(block.get("max_neighbors_offset", 4)),
        )
    rows = run_phase_grid(
        args.output_dir,
        args.model,
        args.temperatures,
        workdir=args.workdir,
        prefix=args.prefix,
        config=args.config,
        replicas=args.replicas,
        cluster=args.cluster,
        backend=backend,
        device=args.device,
        fmax=args.fmax,
        steps=args.steps,
        stages=args.stages,
        symprec=args.symprec,
        reuse=not args.no_reuse,
        **jax_kw,
    )
    for r in rows:
        sgs = "  ".join(str(v) for k, v in r.items() if k.startswith("sg_"))
        print(
            f"P={r['P']:6.2f} GPa  T={r['T']:6.0f} K  V={r['V']:6.2f}  H={r['H']:9.4f}  {sgs}"
        )


if __name__ == "__main__":
    main()
