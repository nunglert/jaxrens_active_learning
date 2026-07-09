"""Tests for the run-side reference-resource resolver.

The resolver's mechanics -- collect needs, build each distinct one once, deliver
back -- are exercised with a fake kind builder (no DFT, no remote_handler). The
registered ``eos`` kind's *reuse* branch (return existing files without
rebuilding) is unit-tested directly; its full build path is integration-covered
in ``test_processors``.
"""

from pathlib import Path

import pytest

from jaxrens_al.panels.reference import ReferenceConsumer, ReferenceNeed
from jaxrens_al.reference_resolver import (
    _build_eos_reference,
    register_reference_kind,
    resolve_references,
)

pytestmark = pytest.mark.local


class _FakePanel(ReferenceConsumer):
    def __init__(self, need):
        self._need = need
        self.received = None

    def reference_needs(self):
        return [self._need]

    def receive_references(self, mapping):
        self.received = mapping[self._need]


def test_resolve_references_dedups_and_delivers(tmp_path):
    calls = []

    def _fake_builder(params, *, evaluator, out_dir, reuse):
        calls.append(params)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return [Path(out_dir) / "produced.extxyz"]

    register_reference_kind("faketest", _fake_builder)

    need = ReferenceNeed.make("faketest", {"element": "Sn"})
    p1, p2 = _FakePanel(need), _FakePanel(need)  # identical need

    delivered = resolve_references(
        [p1, p2], evaluator=object(), workdir=tmp_path
    )

    # Identical needs build exactly once, both panels receive the result.
    assert len(calls) == 1
    assert p1.received == p2.received == delivered[need]


def test_resolve_references_no_needs_is_noop(tmp_path):
    class _Plain:
        pass

    assert (
        resolve_references([_Plain()], evaluator=None, workdir=tmp_path) == {}
    )


def test_resolve_references_unknown_kind_raises(tmp_path):
    need = ReferenceNeed.make("nope-unregistered", {"element": "Sn"})
    with pytest.raises(ValueError, match="No reference builder"):
        resolve_references(
            [_FakePanel(need)], evaluator=None, workdir=tmp_path
        )


def _fake_eos_build(monkeypatch):
    """Patch the (DFT) build so reuse branches are testable with no cluster."""
    calls = []

    def _build(evaluator, *, element, vol_scalings, target_cutoff, out_dir):
        calls.append(element)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = [out / "eos_mp-149.extxyz", out / "eos_mp-165.extxyz"]
        for f in files:
            f.write_text("")
        return files

    monkeypatch.setattr(
        "jaxrens_al.reference_eval.build_eos_reference", _build
    )
    return calls


def test_eos_reference_reuse_returns_existing(tmp_path, monkeypatch):
    # A complete dir (files + manifest sentinel) -> reuse skips the DFT build.
    calls = _fake_eos_build(monkeypatch)
    _build_eos_reference(
        {"element": "Sn"}, evaluator=None, out_dir=tmp_path, reuse=True
    )
    assert calls == ["Sn"]  # first call builds and writes the manifest

    files = _build_eos_reference(
        {"element": "Sn"}, evaluator=None, out_dir=tmp_path, reuse=True
    )
    assert calls == ["Sn"]  # second call reused -- no rebuild
    assert {f.name for f in files} == {
        "eos_mp-149.extxyz",
        "eos_mp-165.extxyz",
    }


def test_eos_reference_partial_dir_without_manifest_rebuilds(
    tmp_path, monkeypatch
):
    # Files on disk but NO manifest (a crashed DFT run) -> rebuild, not reuse.
    calls = _fake_eos_build(monkeypatch)
    (tmp_path / "eos_mp-149.extxyz").write_text("")  # partial, no manifest
    _build_eos_reference(
        {"element": "Sn"}, evaluator=None, out_dir=tmp_path, reuse=True
    )
    assert calls == ["Sn"]  # rebuilt despite the stray file
