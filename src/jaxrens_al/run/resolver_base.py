"""Shared skeleton for the run-side *needs* resolvers.

Both :mod:`jaxrens_al.run.processing_resolver` and
:mod:`jaxrens_al.run.reference_resolver` have the same shape: collect a dashboard's
panels' declared *needs*, look up how each need's ``kind`` is produced in a
registry, derive a short stable tag per distinct step, and deliver the results
back to the panels. Those four mechanics live here so the two resolvers share
one implementation and only their (genuinely different) execution bodies -- a
sharded ``RemoteTrajectoryProcessor`` job vs. a per-need
``GroundTruthEvaluator`` build -- stay in each module.

Pure/neutral: no ``remote_handler`` / jaxrens import, so it is safe on any
import path.
"""

from __future__ import annotations

import hashlib


class KindRegistry:
    """A ``kind -> builder`` map with a uniform unknown-kind error.

    ``label`` names the resolver (``"processor"`` / ``"reference"``) so the
    error message reads naturally for each. The stored builders have
    resolver-specific signatures; this only owns registration + lookup.
    """

    def __init__(self, label: str):
        self._label = label
        self._builders: dict = {}

    def register(self, kind: str, builder) -> None:
        self._builders[kind] = builder

    def builder(self, kind: str):
        """The builder registered for ``kind``, or a uniform ``ValueError``."""
        try:
            return self._builders[kind]
        except KeyError:
            raise ValueError(
                f"No {self._label} builder registered for kind {kind!r}; "
                f"known: {sorted(self._builders)}."
            )

    def __contains__(self, kind) -> bool:
        return kind in self._builders


def collect_needs(panels, needs_attr: str):
    """``(consumers, needs)`` a dashboard's panels declare via ``needs_attr``.

    Consumers are the panels exposing ``needs_attr`` (a ``*_needs()`` method);
    ``needs`` flattens every need they declare, in panel order.
    """
    consumers = [p for p in panels if hasattr(p, needs_attr)]
    needs = [n for p in consumers for n in getattr(p, needs_attr)()]
    return consumers, needs


def deliver(consumers, receive_attr: str, mapping: dict) -> None:
    """Hand ``mapping`` back to each consumer via its ``receive_attr``."""
    for panel in consumers:
        getattr(panel, receive_attr)(mapping)


def stable_tag(prefix: str, key) -> str:
    """``prefix`` + the first 10 hex chars of ``sha1(repr(key))``.

    Stable across runs (so an identical need reuses its on-disk output) and
    keyed by ``key`` (so a different pipeline writes to a different tag).
    """
    return prefix + hashlib.sha1(repr(key).encode()).hexdigest()[:10]
