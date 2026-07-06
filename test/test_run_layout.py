"""Tests for the run-layout bridge (jaxrens_al.run_layout).

This layer only *reads* a run's on-disk artifacts (AL_progress.json + the
calchome tree); it imports neither alviz nor remote_handler. Tests assert the
path resolution and the remote_handler/NeuralIL directory conventions, plus the
defensive globbing (missing dirs -> empty, not an error).
"""

import json

import pytest

from jaxrens_al.run_layout import (
    evaluated_configs,
    evaluated_configs_per_iteration,
    iteration_dir,
    iteration_dirs,
    load_progress,
    ns_result_dir,
    ns_result_dirs,
    sampled_configs,
    sampled_configs_per_iteration,
    training_error_for_iteration,
    training_npz_files,
    training_npz_for_iteration,
)

pytestmark = pytest.mark.local


def test_load_progress_returns_iteration_list(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2)
    progress = load_progress(calchome)
    assert isinstance(progress, list)
    assert len(progress) == 2


def test_load_progress_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_progress(tmp_path)


def test_load_progress_tolerates_dict_keyed_by_iteration(tmp_path):
    calchome = tmp_path / "calchome"
    calchome.mkdir()
    # A dict keyed by iteration index must normalize to an ordered list.
    (calchome / "AL_progress.json").write_text(
        json.dumps({"1": {"steps": [1]}, "0": {"steps": [0]}})
    )
    progress = load_progress(calchome)
    assert [p["steps"] for p in progress] == [[0], [1]]


def test_load_progress_normalizes_staged_schema(make_calchome, tmp_path):
    # The staged StagedActiveLearningHandler schema normalizes to the same
    # ordered per-iteration list as the classic one.
    calchome = make_calchome(tmp_path, n_iter=2, n_sampled=[0, 5], staged=True)
    progress = load_progress(calchome)
    assert isinstance(progress, list) and len(progress) == 2
    # iter 0 had no sampler output; both iters have an evaluator output.
    assert progress[0]["sampler_output"] == []
    assert len(progress[1]["sampler_output"]) == 1
    assert all(len(p["evaluator_output"]) == 1 for p in progress)


def test_staged_and_classic_accessors_agree(make_calchome, tmp_path):
    # Same run, both schemas -> identical resolved paths (rel. to calchome).
    classic = make_calchome(tmp_path / "c", n_iter=2, n_sampled=[0, 5])
    staged = make_calchome(
        tmp_path / "s", n_iter=2, n_sampled=[0, 5], staged=True
    )

    def rels(paths, root):
        return [p.relative_to(root.resolve()) for p in paths]

    assert rels(evaluated_configs(classic), classic) == rels(
        evaluated_configs(staged), staged
    )
    assert rels(sampled_configs(classic), classic) == rels(
        sampled_configs(staged), staged
    )


def test_selected_configs_reads_selector_stage(tmp_path):
    from jaxrens_al.run_layout import (
        selected_configs,
        selected_configs_per_iteration,
    )

    calchome = tmp_path / "calchome"
    (calchome / "selector_output").mkdir(parents=True)
    (calchome / "selector_output" / "sel0.extxyz").write_text("")
    (calchome / "AL_progress.json").write_text(
        json.dumps(
            {
                "stages": ["sampler", "selector", "evaluator", "trainer"],
                "num_iterations": 1,
                "outputs": {
                    "selector": {"0": ["selector_output/sel0.extxyz"]}
                },
            }
        )
    )
    sel = selected_configs(calchome)
    assert len(sel) == 1 and sel[0].name == "sel0.extxyz"
    assert [len(x) for x in selected_configs_per_iteration(calchome)] == [1]


def test_evaluated_and_sampled_configs_in_order(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=3)
    evald = evaluated_configs(calchome)
    sampled = sampled_configs(calchome)
    assert len(evald) == 3 and len(sampled) == 3
    assert all(p.is_absolute() and p.exists() for p in evald + sampled)
    # Order follows AL_progress (iteration 0, 1, 2).
    assert [p.parent.name for p in evald] == ["evaluator_output"] * 3


def test_per_iteration_accessors_align_with_progress(make_calchome, tmp_path):
    # Iteration 0 has no sampler output; both have evaluated configs.
    calchome = make_calchome(tmp_path, n_iter=2, n_sampled=[0, 5])
    ev = evaluated_configs_per_iteration(calchome)
    sa = sampled_configs_per_iteration(calchome)
    assert [len(x) for x in ev] == [1, 1]
    assert [len(x) for x in sa] == [0, 1]  # iter 0 produced no samples


def test_training_accessors_use_training_results_convention(
    make_calchome, tmp_path
):
    # The NeuralIL trainer writes under <calcdir>/training/results.
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    npz = training_npz_files(calchome)
    assert len(npz) == 2
    assert all(p.match("*/training/results/*.npz") for p in npz)

    it1 = iteration_dir(calchome, 1)
    assert len(training_npz_for_iteration(it1)) == 1
    assert len(training_error_for_iteration(it1)) == 1


def test_training_accessors_empty_when_no_artifacts(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, with_training=False)
    assert training_npz_files(calchome) == []
    assert training_npz_for_iteration(iteration_dir(calchome, 0)) == []


def test_resampling_artifacts_none_when_absent(make_calchome, tmp_path):
    from jaxrens_al.run_layout import (
        quests_pca_json,
        resampled_provenance_json,
    )

    calchome = make_calchome(tmp_path, n_iter=1)
    it0 = iteration_dir(calchome, 0)
    assert quests_pca_json(it0) is None
    assert resampled_provenance_json(it0) is None


def test_resampling_artifacts_found_when_present(make_calchome, tmp_path):
    from jaxrens_al.run_layout import (
        quests_pca_json,
        resampled_provenance_json,
    )

    calchome = make_calchome(tmp_path, n_iter=1)
    results = iteration_dir(calchome, 0) / "ns_sampler" / "results"
    results.mkdir(parents=True)
    (results / "quests_pca.json").write_text("{}")
    (results / "resampled_provenance.json").write_text("{}")

    it0 = iteration_dir(calchome, 0)
    assert quests_pca_json(it0) == results / "quests_pca.json"
    assert (
        resampled_provenance_json(it0) == results / "resampled_provenance.json"
    )


def test_ns_result_dir_none_without_energies(make_calchome, tmp_path):
    # No jaxrens .energies were written, so there is no NS result dir.
    calchome = make_calchome(tmp_path, n_iter=2)
    assert ns_result_dir(iteration_dir(calchome, 0)) is None
    assert ns_result_dirs(calchome) == []


def test_iteration_dirs_numeric_sorted(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    dirs = iteration_dirs(calchome)
    assert [d.name for d in dirs] == ["0", "1"]


def test_training_results_dir_prefers_train_then_training(tmp_path):
    """The trainer working dir is 'train' (staged) or 'training' (older)."""
    from jaxrens_al.run_layout import training_results_dir

    it = tmp_path / "0"
    assert training_results_dir(it) is None  # neither present

    (it / "training" / "results").mkdir(parents=True)
    assert training_results_dir(it) == it / "training" / "results"

    # When both exist, 'train' (the staged handler's name) wins.
    (it / "train" / "results").mkdir(parents=True)
    assert training_results_dir(it) == it / "train" / "results"


def test_training_accessors_read_the_train_dir(tmp_path, write_mace_train_log):
    """MACE logs under <i>/train/results are found (staged layout)."""
    from jaxrens_al.run_layout import training_mace_for_iteration

    res = tmp_path / "0" / "train" / "results" / "calcdirs" / "0" / "results"
    res.mkdir(parents=True)
    write_mace_train_log(res / "Model_run-1_train.txt")
    hits = training_mace_for_iteration(tmp_path / "0")
    assert len(hits) == 1 and hits[0].name == "Model_run-1_train.txt"


def test_detect_ns_prefix_reads_mace_prefix(tmp_path):
    """The prefix is sniffed off <prefix>.run*.energies (e.g. mace_sn16)."""
    from jaxrens_al.run_layout import detect_ns_prefix

    out = tmp_path / "0" / "ns_sampler" / "results" / "output"
    out.mkdir(parents=True)
    for r in (0, 1):
        (out / f"mace_sn16.run0{r}.energies").write_text("# stub\n")
    assert detect_ns_prefix(tmp_path / "0") == "mace_sn16"


def test_detect_ns_prefix_default_when_absent(tmp_path):
    from jaxrens_al.run_layout import detect_ns_prefix

    assert detect_ns_prefix(tmp_path / "0", default="ns") == "ns"
