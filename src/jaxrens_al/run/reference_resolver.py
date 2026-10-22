"""Resolve a dashboard's *reference-resource* needs in one run-side pass.

The sibling of :mod:`jaxrens_al.run.processing_resolver`. A panel that needs a
produced reference artifact declares a
:class:`~jaxrens_al.dashboard.panels.reference.ReferenceNeed`; this module collects those
needs across a dashboard, builds each distinct one once (with on-disk reuse),
and delivers the produced files back via ``receive_references``. So the build
orchestration lives in one place, not wired into each panel.

Why a separate resolver from ``resolve_processing``: a reference is produced by
*DFT evaluation* (a ``GroundTruthEvaluator``), not a trajectory ``Processor``,
so the executor is a configured **evaluator**, not a cluster context. The EOS
reference is the first (and only) registered kind. Because it needs a DFT
evaluator (cluster + INCAR), :func:`resolve_references` is called from a
**run-side** dashboard driver that builds that evaluator -- not the light
``jaxrens-al dashboard`` CLI, where an ``eos`` entry with explicit
``references:`` is the no-evaluator path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jaxrens_al.run.resolver_base import (
    KindRegistry,
    collect_needs,
    deliver,
    stable_tag,
)

logger = logging.getLogger(__name__)

# kind -> builder(params_dict, *, evaluator, out_dir, reuse) -> list[Path]
_REGISTRY = KindRegistry("reference")
# kind -> whether its builder needs a (DFT) evaluator. Evaluator-free kinds
# (e.g. an MP structure fetch) can be resolved by the light dashboard CLI,
# which has no evaluator; evaluator-needing kinds require a run-side driver.
_NEEDS_EVALUATOR: dict = {}


def register_reference_kind(
    kind: str, builder, *, needs_evaluator: bool = True
) -> None:
    """Register how a :class:`ReferenceNeed` of ``kind`` is produced.

    ``builder(params, *, evaluator, out_dir, reuse) -> list[Path]`` runs the
    production for one distinct need, writing into ``out_dir`` and honouring
    ``reuse`` (return existing files without rebuilding). Called run-side (may
    import ``remote_handler`` / the upstream). Set ``needs_evaluator=False`` for
    a builder that produces structures only (no DFT), so it is resolvable
    without an evaluator -- e.g. from the light ``jaxrens-al dashboard`` CLI.
    """
    _REGISTRY.register(kind, builder)
    _NEEDS_EVALUATOR[kind] = needs_evaluator


def resolve_references(
    panels, *, evaluator, workdir, reuse: bool = True
) -> dict:
    """Produce the reference artifacts a dashboard's panels need, once each.

    Collects every panel's ``reference_needs``, builds each distinct need with
    its registered kind builder (deduplicated; each need's output keyed into its
    own dir under ``workdir`` so an identical re-render reuses it), and delivers
    ``{ReferenceNeed -> [files]}`` back to each panel via ``receive_references``.

    Args:
        panels: The dashboard's panels (those without needs are ignored).
        evaluator: A configured ``GroundTruthEvaluator`` (the DFT level).
        workdir: Root dir the per-need output dirs are created under.
        reuse: Skip a build when its output files already exist.

    Returns:
        The ``{need -> [files]}`` mapping (empty if no needs).
    """
    consumers, needs = collect_needs(panels, "reference_needs")
    if not needs:
        return {}

    workdir = Path(workdir)
    delivered: dict = {}
    for need in needs:
        if need in delivered:  # identical needs build once
            continue
        builder = _REGISTRY.builder(need.kind)  # raises on unknown kind
        if evaluator is None and _NEEDS_EVALUATOR.get(need.kind, True):
            # No evaluator here (e.g. the light dashboard CLI) and this kind
            # needs one -> skip; its consumers fall back / self-report. Run it
            # from a run-side driver that builds the evaluator.
            logger.warning(
                "resolve_references: skipping %s need %s -- it needs a DFT "
                "evaluator; run from a run-side driver.",
                need.kind,
                dict(need.params),
            )
            continue
        tag = stable_tag(need.kind + "_", (need.kind, need.params))
        out_dir = workdir / tag
        logger.info(
            "resolve_references: %s need %s -> %s",
            need.kind,
            dict(need.params),
            out_dir,
        )
        delivered[need] = builder(
            need.params_dict, evaluator=evaluator, out_dir=out_dir, reuse=reuse
        )

    deliver(consumers, "receive_references", delivered)
    return delivered


def _build_eos_reference(params: dict, *, evaluator, out_dir, reuse: bool):
    """Produce (or reuse) one element's per-phase EOS reference scans.

    Reuse is gated on a ``.eos_manifest.json`` sentinel written only after a
    *complete* build (listing every produced file), so a DFT run that crashed
    part-way -- some ``eos_*.extxyz`` on disk but no manifest -- is rebuilt
    rather than mistaken for finished.
    """
    out_dir = Path(out_dir)
    manifest = out_dir / ".eos_manifest.json"
    if reuse and manifest.exists():
        listed = [out_dir / name for name in json.loads(manifest.read_text())]
        if listed and all(p.exists() for p in listed):
            logger.info(
                "resolve_references: reusing %d EOS file(s) in %s",
                len(listed),
                out_dir,
            )
            return listed

    from jaxrens_al.run.reference_db import mp_query_kwargs, resolve_system
    from jaxrens_al.run.reference_eval import build_eos_reference

    # The EOS panel works per atom and rebuilds a cutoff-hosting supercell at
    # model-eval time, so the DFT reference is labelled on the plain unit cells
    # (cheaper -- fewer atoms). ``target_cutoff`` in the need is the panel's
    # model-expansion cutoff, not a DFT supercell size; keep it out of the build
    # (it still keys the build for reuse via the need's params).
    files = build_eos_reference(
        evaluator,
        system=resolve_system(params.get("system"), params.get("element")),
        vol_scalings=params.get("vol_scalings"),
        target_cutoff=None,
        out_dir=out_dir,
        **mp_query_kwargs(params),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([p.name for p in files]))
    return files


register_reference_kind("eos", _build_eos_reference)


def _build_mp_reference(params: dict, *, evaluator, out_dir, reuse: bool):
    """Produce (or reuse) an MP-drawn reference config set.

    The reference the atomic-environments PCA is fit on: Materials-Project
    phases of the requested system + a rattle grid
    (:func:`build_reference_database`). The need's ``element``/``system``
    param may name an element, a formula or a chemsys, and may carry any
    ``MP_QUERY_KEYS`` filter (``max_e_above_hull``, ...). Needs
    **no** DFT evaluator (structures only, not labelled energies), so
    ``evaluator`` is ignored -- it just wants the ``[refdb]`` extra + network.
    Reuse is gated on a ``.mp_manifest.json`` sentinel written only after a
    complete build, so a fetch that crashed part-way is rebuilt.
    """
    out_dir = Path(out_dir)
    manifest = out_dir / ".mp_manifest.json"
    if reuse and manifest.exists():
        listed = [out_dir / name for name in json.loads(manifest.read_text())]
        if listed and all(p.exists() for p in listed):
            logger.info(
                "resolve_references: reusing MP reference in %s", out_dir
            )
            return listed

    from jaxrens_al.run.reference_db import (
        build_reference_database,
        mp_query_kwargs,
        resolve_system,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reference.extxyz"
    build_reference_database(
        resolve_system(params.get("system"), params.get("element")),
        target_cutoff=params.get("target_cutoff", 3.0),
        output=out_file,
        **mp_query_kwargs(params),
    )
    manifest.write_text(json.dumps([out_file.name]))
    return [out_file]


register_reference_kind(
    "mp_elemental", _build_mp_reference, needs_evaluator=False
)
# Same builder under a system-agnostic name (the kind is no longer elemental);
# ``mp_elemental`` stays registered so existing needs keep resolving.
register_reference_kind("mp", _build_mp_reference, needs_evaluator=False)


__all__ = [
    "resolve_references",
    "register_reference_kind",
]
