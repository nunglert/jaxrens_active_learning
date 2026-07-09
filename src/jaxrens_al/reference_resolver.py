"""Resolve a dashboard's *reference-resource* needs in one run-side pass.

The sibling of :mod:`jaxrens_al.processing_resolver`. A panel that needs a
produced reference artifact declares a
:class:`~jaxrens_al.panels.reference.ReferenceNeed`; this module collects those
needs across a dashboard, builds each distinct one once (with on-disk reuse),
and delivers the produced files back via ``receive_references``. So the build
orchestration lives in one place, not wired into each panel.

Why a separate resolver from ``resolve_processing``: a reference is produced by
*DFT evaluation* (a ``GroundTruthEvaluator``), not a trajectory ``Processor``,
so the executor is a configured **evaluator**, not a cluster context. The EOS
reference is the first (and only) registered kind. Because it needs a DFT
evaluator (cluster + INCAR), :func:`resolve_references` is called from a
**run-side** dashboard driver that builds that evaluator -- not the light
``jaxrens-al-dashboard`` CLI, whose ``--eos-reference`` flag stays the
no-evaluator fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jaxrens_al.resolver_base import (
    KindRegistry,
    collect_needs,
    deliver,
    stable_tag,
)

logger = logging.getLogger(__name__)

# kind -> builder(params_dict, *, evaluator, out_dir, reuse) -> list[Path]
_REGISTRY = KindRegistry("reference")


def register_reference_kind(kind: str, builder) -> None:
    """Register how a :class:`ReferenceNeed` of ``kind`` is produced.

    ``builder(params, *, evaluator, out_dir, reuse) -> list[Path]`` runs the
    (DFT) production for one distinct need, writing into ``out_dir`` and
    honouring ``reuse`` (return existing files without rebuilding). Called
    run-side (may import ``remote_handler`` / the upstream).
    """
    _REGISTRY.register(kind, builder)


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

    from jaxrens_al.reference_eval import build_eos_reference

    files = build_eos_reference(
        evaluator,
        element=params["element"],
        vol_scalings=params.get("vol_scalings"),
        target_cutoff=params.get("target_cutoff", 4.0),
        out_dir=out_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([p.name for p in files]))
    return files


register_reference_kind("eos", _build_eos_reference)


__all__ = [
    "resolve_references",
    "register_reference_kind",
]
