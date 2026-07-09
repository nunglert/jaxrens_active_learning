"""Panel processing requirements -- plain data, resolved run-side.

A producer-backed panel does not annotate anything itself. It *declares* what
processing its input trajectory needs (a :class:`ProcessingNeed`) and later
*receives* the resolved annotated path. Both halves are pure data / attribute
assignment -- no ``remote_handler``, no cluster -- so panels stay light and
dashboard-side. The run-side :mod:`jaxrens_al.run.processing_resolver` is what
collects needs across a dashboard, runs a single processing job, and delivers
the results back.

This is the seam that keeps the "annotate the trajectory" orchestration in one
place instead of wiring a producer into every panel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingNeed:
    """A panel's request to have a trajectory annotated by one processor step.

    Purely declarative and hashable, so identical needs across panels
    deduplicate to one processor step in the resolved pipeline.

    Attributes:
        inputs: Trajectory file(s) to annotate -- the grouping key.
        kind: Processor registry kind (e.g. ``"steinhardt"``), mapped to a
            concrete ``Processor`` by the run-side resolver's builder registry.
        params: Sorted ``(key, value)`` pairs of the processor's parameters
            (a tuple so the need stays hashable / dedupable).
    """

    inputs: tuple
    kind: str
    params: tuple

    @classmethod
    def make(cls, inputs, kind: str, params: dict) -> "ProcessingNeed":
        """Build a need, canonicalising ``inputs``/``params`` for hashing."""
        canon = tuple(sorted((k, _hashable(v)) for k, v in params.items()))
        return cls(tuple(str(i) for i in inputs), kind, canon)

    @property
    def params_dict(self) -> dict:
        return {k: v for k, v in self.params}


def _hashable(v):
    """Make list-valued params hashable (e.g. ``l_values``)."""
    if isinstance(v, list):
        return tuple(v)
    return v


class ProcessingConsumer:
    """Mixin for a panel that needs a run-side processing step.

    Override :meth:`processing_needs` to declare what annotation the panel's
    input trajectory requires, and :meth:`receive_annotations` to pick up the
    resolved artifact path(s). A panel handed an already-annotated file
    declares nothing (default empty needs) and is read directly.
    """

    def processing_needs(self) -> list:
        """The processing this panel needs (default: none)."""
        return []

    def receive_annotations(self, mapping: dict) -> None:
        """Receive ``{input_file -> annotated_file}`` from the resolver."""
