"""Run-side producer: a DFT-labelled EOS reference for the EOS panel.

The reference-resource counterpart to :mod:`jaxrens_al.processors.steinhardt`.
Steinhardt is a ``remote_handler`` trajectory ``Processor`` (shard analysis);
this is **not** -- it *produces* a new artifact by DFT evaluation, the resource
the EOS panel declares a :class:`~jaxrens_al.panels.reference.ReferenceNeed`
for. It lives in ``processors`` as the run-side home for panel producers, but
its execution machinery is a ``GroundTruthEvaluator`` (VASP/ORCA), passed in
already configured, so this module imports **no** ``remote_handler`` itself.

:func:`build_eos_reference` fetches an element's phases (Materials Project),
builds a zero-rattle volume scan per phase (:func:`~jaxrens_al.reference_db.
eos_phase_scans`), DFT-labels each with the given evaluator
(:func:`~jaxrens_al.reference_eval.evaluate_config_sets`), and publishes one
``eos_<phase>.extxyz`` per phase -- exactly what the EOS panel reads.
"""

from __future__ import annotations

from pathlib import Path


def build_eos_reference(
    evaluator,
    *,
    element: str,
    vol_scalings=None,
    target_cutoff: float = 4.0,
    out_dir,
    workdir=None,
    api_key: str | None = None,
) -> list[Path]:
    """Fetch, volume-scan, and DFT-label one element's EOS reference phases.

    Args:
        evaluator: A configured ``GroundTruthEvaluator`` (duck-typed; reuse the
            one from your AL loop so the DFT level matches).
        element: Chemical symbol to build the reference for.
        vol_scalings: Cell-vector scale factors for the scan (default: the
            ``eos_phase_scans`` grid).
        target_cutoff: Supercell cutoff (Å) -- set to the model's ``r_cut`` so
            the EOS panel's model re-score is exact (jaxrens uses a
            minimum-image neighbour list valid only when the cell hosts it).
        out_dir: Where the per-phase ``eos_<phase>.extxyz`` files are written.
        workdir: Scratch dir for the evaluator's per-phase calc dirs (default:
            ``out_dir/work``).
        api_key: Materials Project key (default: ``$MP_API_KEY``).

    Returns:
        The per-phase labelled scan paths, one ``eos_<phase>.extxyz`` each.
    """
    import ase.io

    from jaxrens_al.reference_db import eos_phase_scans, fetch_mp_elemental
    from jaxrens_al.reference_eval import evaluate_config_sets

    out_dir = Path(out_dir)
    workdir = Path(workdir) if workdir is not None else out_dir / "work"

    phases = fetch_mp_elemental(element, api_key=api_key)
    scans = eos_phase_scans(
        phases, vol_scalings=vol_scalings, target_cutoff=target_cutoff
    )
    sets = {f"eos_{label}": scan for label, scan in scans.items()}
    labelled = evaluate_config_sets(evaluator, sets, workdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for label, path in labelled.items():
        dest = out_dir / f"{label}.extxyz"
        ase.io.write(str(dest), ase.io.read(str(path), index=":"))
        files.append(dest)
    return files
