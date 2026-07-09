"""Panel *reference-resource* requirements -- plain data, resolved run-side.

The sibling of :mod:`jaxrens_al.dashboard.panels.processing`, for a panel that needs a
whole reference *artifact produced for it* rather than an existing trajectory
annotated. The EOS panel is the first consumer: it needs a DFT-labelled,
per-phase volume scan it cannot itself compute (Materials-Project fetch +
supercell scan + DFT single-points). Instead of the user building that by hand
and passing files, the panel *declares* what reference it needs (a
:class:`ReferenceNeed`) and later *receives* the produced files.

The distinction from ``processing`` is deliberate: a ``ProcessingNeed`` is keyed
by the *input trajectory* it annotates and is fulfilled by a ``remote_handler``
trajectory ``Processor`` (shard analysis). A ``ReferenceNeed`` has **no** input
file -- its identity is its parameters (element, DFT level, volume grid) -- and
is fulfilled by *DFT evaluation* (a different ``remote_handler`` subsystem). So
the two get parallel need/consumer types and parallel run-side resolvers
(:func:`jaxrens_al.run.reference_resolver.resolve_references`), while both keep the
panel pure data / attribute assignment (no ``remote_handler``, dashboard-side).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceNeed:
    """A panel's request for a produced reference artifact.

    Purely declarative and hashable, so identical needs across panels
    deduplicate to a single build in the resolved pipeline (and a stable key
    for on-disk reuse across re-renders).

    Attributes:
        kind: Reference registry kind (e.g. ``"eos"``), mapped to a concrete
            producer by the run-side resolver's builder registry.
        params: Sorted ``(key, value)`` pairs identifying the resource (a tuple
            so the need stays hashable / dedupable).
    """

    kind: str
    params: tuple

    @classmethod
    def make(cls, kind: str, params: dict) -> "ReferenceNeed":
        """Build a need, canonicalising ``params`` for hashing."""
        canon = tuple(sorted((k, _hashable(v)) for k, v in params.items()))
        return cls(kind, canon)

    @property
    def params_dict(self) -> dict:
        return {k: v for k, v in self.params}


def _hashable(v):
    """Make list-valued params hashable (e.g. ``vol_scalings``)."""
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v


class ReferenceConsumer:
    """Mixin for a panel that needs a run-side-produced reference artifact.

    Override :meth:`reference_needs` to declare what reference the panel
    requires, and :meth:`receive_references` to pick up the produced file(s). A
    panel handed explicit files declares nothing (default empty needs) and reads
    them directly.
    """

    def reference_needs(self) -> list:
        """The reference artifacts this panel needs (default: none)."""
        return []

    def receive_references(self, mapping: dict) -> None:
        """Receive ``{ReferenceNeed -> [produced files]}`` from the resolver."""
