"""Project panels for the training stage.

Four panels read what a trainer leaves on disk:

* :class:`NeuralILParityPanel` -- equips alviz's abstract ``ParityPanel``
  (rendering + ``ParityData`` contract) with computation: it reads the
  ``save_data`` ``.npz`` bundles (``e_obs``/``e_pred``/``f_obs``/``f_pred``)
  the trainer writes, instead of re-scoring a model. alviz owns the plot; we
  own *where the numbers come from*.
* :class:`ModelParityPanel` -- also a ``ParityPanel``, but it *re-scores* a
  labelled set with the freshly trained model through the jaxrens backend
  (the same seam the EOS panel uses), so it works for a backend that writes no
  parity bundle -- notably **MACE**. Predicted energies/forces vs the DFT
  reference (``REF_energy``/``REF_forces``). This is the "how good is the model
  we just trained" plot, distinct from the sampler-surrogate parity that
  ``alviz.ExtxyzParityPanel`` gives on ``ns_energy``.
* :class:`NeuralILTrainingCurve` -- a from-scratch ``alviz.Entry`` (alviz has
  no training-curve skeleton): parses the per-epoch validation RMSE out of the
  NeuralIL trainer's ``error_*.txt`` log and plots E/F RMSE vs epoch.
* :class:`MaceTrainingCurve` -- the same idea for a MACE(-finetuning) run,
  parsing the per-epoch validation metrics out of MACE's JSON-lines
  ``*_train.txt`` log (the ``mode == "eval"`` rows).
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# Panel specs.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, PanelSpec, register_panel, register_simple
from alviz.panels import ParityData, ParityPanel

from jaxrens_al.dashboard.panels.model_eval import (
    ModelEvalConsumer,
    ModelEvalNeed,
    model_eval_keys,
)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


class NeuralILParityPanel(ParityPanel):
    """Energy/force parity from NeuralIL ``save_data`` ``.npz`` bundles.

    Each npz holds ``e_obs``/``e_pred`` (per-config totals) and
    ``f_obs``/``f_pred`` (per-atom force arrays). With ``per_atom=True``
    energies are divided by atom count, recovered from the ``types`` array the
    bundle also stores (falling back to no normalization if absent).

    Args:
        npz_files: One or more ``.npz`` paths (e.g. the validation bundle of
            each iteration). Empty -> the panel has nothing to plot.
    """

    name = "neuralil_parity"

    def __init__(
        self,
        npz_files,
        *args,
        per_atom: bool = True,
        show_forces: bool = True,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            show_forces=show_forces,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )
        self._npz_files = [str(p) for p in npz_files]
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> ParityData:
        e_obs, e_pred = [], []
        f_obs, f_pred = [], []

        for path in self._npz_files:
            with np.load(path, allow_pickle=True) as data:
                eo = np.asarray(data["e_obs"], dtype=float).ravel()
                ep = np.asarray(data["e_pred"], dtype=float).ravel()
                norm = self._atom_counts(data, len(eo))
                e_obs.extend((eo / norm).tolist())
                e_pred.extend((ep / norm).tolist())
                if self._show_forces and "f_obs" in data:
                    f_obs.extend(
                        np.asarray(data["f_obs"], dtype=float).ravel()
                    )
                    f_pred.extend(
                        np.asarray(data["f_pred"], dtype=float).ravel()
                    )

        e_obs = np.asarray(e_obs, dtype=float)
        e_pred = np.asarray(e_pred, dtype=float)
        fields = dict(
            e_obs=e_obs.tolist(),
            e_pred=e_pred.tolist(),
            e_rmse=_rmse(e_obs, e_pred),
            e_mae=_mae(e_obs, e_pred),
            per_atom=self._per_atom,
        )
        if self._show_forces and f_obs:
            f_obs = np.asarray(f_obs, dtype=float)
            f_pred = np.asarray(f_pred, dtype=float)
            fields.update(
                f_obs=f_obs.tolist(),
                f_pred=f_pred.tolist(),
                f_rmse=_rmse(f_obs, f_pred),
                f_mae=_mae(f_obs, f_pred),
            )
        return ParityData(**fields)

    def _atom_counts(self, data, n_configs):
        """Per-config atom counts for per-atom normalization (else ones)."""
        if not self._per_atom or "types" not in data:
            return np.ones(n_configs, dtype=float)
        types = data["types"]
        # ``types`` is one per-atom array per config (object array) or a
        # single 2D (n_config, n_atoms) block.
        try:
            counts = np.array([len(t) for t in types], dtype=float)
        except TypeError:
            counts = np.full(n_configs, float(np.shape(types)[-1]))
        if counts.shape[0] != n_configs:
            return np.ones(n_configs, dtype=float)
        return counts


class ModelParityPanel(ModelEvalConsumer, ParityPanel):
    """Trained-model predictions vs DFT reference, by *re-scoring* a set.

    Loads the freshly trained model through the jaxrens backend (mirroring the
    EOS panel's model seam) and evaluates each configuration, so it needs no
    parity bundle from the trainer -- the case for **MACE**, which writes none.
    (For NeuralIL prefer :class:`NeuralILParityPanel`, which reads the
    already-computed ``.npz`` bundle instead of re-running the model.)

    Like the EOS panel, the model re-score runs either in-process (CPU
    backends) or -- when ``remote_eval`` -- on a GPU cluster via the shared
    ``model_eval`` processor (required for a MACE-jax bundle); the predicted
    energy/forces are then read back from the annotated config files.

    Args:
        config_files: One or more labelled extxyz paths (e.g. the validation
            split ``db_val.extxyz``) carrying ``REF_energy`` / ``REF_forces``.
        model_file: The trained-model bundle to load (mace-jax ``.pkl`` /
            neuralil ``.pkl``).
        backend_type: jaxrens backend to load the model with (``mace`` /
            ``neuralil``).
        energy_ref_key / forces_ref_key: info/array keys of the DFT reference.
        remote_eval: Evaluate the model on a GPU cluster instead of in-process.
        max_neighbors: Static neighbor-buffer size for the backend call (the
            starting bucket when ``max_neighbors_list`` is given).
        batch_size: Configs per vmapped backend call (see
            :func:`jaxrens_al.dashboard.panels.model_eval.score_atoms`). Unset
            evaluates one config per call.
        max_neighbors_list: Neighbour-bucket ladder; sizes the buffer from the
            geometry instead of pinning it at ``max_neighbors``. Only used
            together with ``batch_size``.
        max_neighbors_offset: Headroom above the observed neighbour count when
            picking a bucket.
    """

    name = "model_parity"

    def __init__(
        self,
        config_files,
        model_file,
        *args,
        backend_type: str = "neuralil",
        energy_ref_key: str = "REF_energy",
        forces_ref_key: str = "REF_forces",
        per_atom: bool = True,
        show_forces: bool = True,
        remote_eval: bool = False,
        max_neighbors: int = 50,
        batch_size: int | None = None,
        max_neighbors_list=None,
        max_neighbors_offset: int = 4,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            show_forces=show_forces,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )
        self._config_files = [str(p) for p in config_files]
        self._model_file = str(model_file)
        self._backend_type = backend_type
        self._energy_ref_key = energy_ref_key
        self._forces_ref_key = forces_ref_key
        self._per_atom = per_atom
        self._remote_eval = bool(remote_eval)
        self._model_eval_annotated: dict = {}
        self._max_neighbors = max_neighbors
        self._batch_size = batch_size
        self._max_neighbors_list = max_neighbors_list
        self._max_neighbors_offset = max_neighbors_offset

    def model_eval_needs(self) -> list:
        """Score the config set with the model on the cluster (remote mode)."""
        if not self._remote_eval:
            return []
        return [
            ModelEvalNeed.make(
                self._config_files,
                self._model_file,
                self._backend_type,
                want_forces=self._show_forces,
                max_neighbors=self._max_neighbors,
                batch_size=self._batch_size,
                max_neighbors_list=self._max_neighbors_list,
                max_neighbors_offset=self._max_neighbors_offset,
            )
        ]

    def receive_model_eval(self, mapping: dict) -> None:
        if self._remote_eval:
            by_model = mapping.get(self._model_file, {})
            self._model_eval_annotated = {
                f: by_model[f] for f in self._config_files if f in by_model
            }

    def _evaluate(self, model_file, atoms_list):
        """Predicted totals under the model (seam; patched in tests).

        Returns ``(energies, forces)``: ``energies`` a list of total energies
        (eV); ``forces`` a list of ``(n_atoms, 3)`` arrays, or ``None`` when
        ``show_forces`` is off.
        """
        from jaxrens_al.dashboard.panels.model_eval import score_atoms

        return score_atoms(
            model_file,
            self._backend_type,
            atoms_list,
            want_forces=self._show_forces,
            max_neighbors=self._max_neighbors,
            batch_size=self._batch_size,
            max_neighbors_list=self._max_neighbors_list,
            max_neighbors_offset=self._max_neighbors_offset,
        )

    def _predict_remote(self, frames):
        """Read model energies/forces the cluster wrote onto the frames."""
        energy_key, forces_key = model_eval_keys(self._model_file)
        energies = [float(at.info[energy_key]) for at in frames]
        forces = None
        if self._show_forces:
            forces = [
                np.asarray(at.arrays[forces_key], dtype=float) for at in frames
            ]
        return energies, forces

    def compute_data(self, shared_ressources: dict) -> ParityData:
        import ase.io

        # In remote mode read the annotated config files (they carry the model
        # predictions + the DFT reference keys); otherwise the raw config files.
        sources = self._config_files
        if self._remote_eval:
            if not self._model_eval_annotated:
                raise RuntimeError(
                    "model_parity is remote_eval but its configs are not "
                    "annotated yet: run resolve_model_evaluations() over the "
                    "dashboard before rendering."
                )
            sources = [
                self._model_eval_annotated[f] for f in self._config_files
            ]

        frames = []
        for path in sources:
            frames.extend(ase.io.read(str(path), index=":"))

        if self._remote_eval:
            e_pred_tot, f_pred_arrays = self._predict_remote(frames)
        else:
            e_pred_tot, f_pred_arrays = self._evaluate(
                self._model_file, frames
            )

        e_obs, e_pred = [], []
        f_obs, f_pred = [], []
        for i, atoms in enumerate(frames):
            norm = len(atoms) if self._per_atom else 1
            e_obs.append(float(atoms.info[self._energy_ref_key]) / norm)
            e_pred.append(float(e_pred_tot[i]) / norm)
            if self._show_forces and f_pred_arrays is not None:
                f_obs.extend(
                    np.asarray(
                        atoms.arrays[self._forces_ref_key], dtype=float
                    ).ravel()
                )
                f_pred.extend(
                    np.asarray(f_pred_arrays[i], dtype=float).ravel()
                )

        e_obs = np.asarray(e_obs, dtype=float)
        e_pred = np.asarray(e_pred, dtype=float)
        fields = dict(
            e_obs=e_obs.tolist(),
            e_pred=e_pred.tolist(),
            e_rmse=_rmse(e_obs, e_pred),
            e_mae=_mae(e_obs, e_pred),
            per_atom=self._per_atom,
        )
        if self._show_forces and f_obs:
            f_obs = np.asarray(f_obs, dtype=float)
            f_pred = np.asarray(f_pred, dtype=float)
            fields.update(
                f_obs=f_obs.tolist(),
                f_pred=f_pred.tolist(),
                f_rmse=_rmse(f_obs, f_pred),
                f_mae=_mae(f_obs, f_pred),
            )
        return ParityData(**fields)


# Pull "i_epoch=<N>" and "E_RMSE = <x>" / "F_RMSE = <x>" out of the trainer's
# error_*.txt (one multi-line block per epoch, possibly several models each).
_RE_EPOCH = re.compile(r"i_epoch\s*=\s*(\d+)")
_RE_ERMSE = re.compile(r"E_RMSE\s*=\s*([-+0-9.eE]+)")
_RE_FRMSE = re.compile(r"F_RMSE\s*=\s*([-+0-9.eE]+)")


class NeuralILTrainingCurve(Entry):
    """Validation RMSE vs epoch, parsed from NeuralIL ``error_*.txt`` logs.

    Several ensemble members may report per epoch; their RMSEs are averaged.
    Multiple log files (one per iteration's training) are concatenated and
    distinguished by a restarting epoch axis.

    Args:
        error_files: One or more ``error_*.txt`` paths. Empty -> nothing to
            plot.
    """

    name = "neuralil_training_curve"

    def __init__(self, error_files, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error_files = [str(p) for p in error_files]

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        epochs, e_rmse, f_rmse = [], [], []
        for path in self._error_files:
            ep, er, fr = self._parse_one(path)
            epochs.extend(ep)
            e_rmse.extend(er)
            f_rmse.extend(fr)
        return JsonDict(
            step=list(range(len(epochs))),
            epoch=epochs,
            e_rmse=e_rmse,
            f_rmse=f_rmse,
        )

    @staticmethod
    def _parse_one(path):
        epochs, e_rmse, f_rmse = [], [], []
        cur_epoch = None
        cur_e, cur_f = [], []

        def flush():
            if cur_epoch is not None and cur_e:
                epochs.append(cur_epoch)
                e_rmse.append(float(np.mean(cur_e)))
                f_rmse.append(float(np.mean(cur_f)) if cur_f else float("nan"))

        with open(path) as fh:
            for line in fh:
                m = _RE_EPOCH.search(line)
                if m:
                    flush()
                    cur_epoch = int(m.group(1))
                    cur_e, cur_f = [], []
                    continue
                me = _RE_ERMSE.search(line)
                if me:
                    cur_e.append(float(me.group(1)))
                mf = _RE_FRMSE.search(line)
                if mf:
                    cur_f.append(float(mf.group(1)))
        flush()
        return epochs, e_rmse, f_rmse

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def plot_data(self):
        step = self.data["step"]
        self.ax.plot(
            step, self.data["e_rmse"], color="tab:blue", label="E RMSE"
        )
        self.ax.set_xlabel("logged step (epoch checkpoints)")
        self.ax.set_ylabel("energy RMSE", color="tab:blue")
        self.ax.set_yscale("log")
        self.ax.set_title("NeuralIL training error")

        f_rmse = np.asarray(self.data["f_rmse"], dtype=float)
        if np.isfinite(f_rmse).any():
            ax2 = self.ax.twinx()
            ax2.plot(step, f_rmse, color="tab:red", label="F RMSE")
            ax2.set_ylabel("force RMSE", color="tab:red")
            ax2.set_yscale("log")


class MaceTrainingCurve(Entry):
    """Validation error vs epoch, parsed from a MACE ``*_train.txt`` log.

    MACE writes one JSON object per line; the ``mode == "eval"`` rows carry the
    per-epoch validation metrics (``rmse_e_per_atom``, ``rmse_f``, ``mae_*``,
    ...). This panel plots an energy and a force metric vs epoch on twin axes
    -- the at-a-glance "did the finetuning converge". Several logs (e.g. one
    per finetuning iteration) are concatenated along a restarting epoch axis.
    Values are in the log's native units (MACE writes errors in meV / meV per
    angstrom).

    Args:
        train_logs: One or more MACE ``*_train.txt`` paths. Empty -> nothing to
            plot.
        energy_metric: Which eval-row key to plot on the left axis (default
            ``rmse_e_per_atom``).
        force_metric: Which eval-row key to plot on the right axis (default
            ``rmse_f``).
    """

    name = "mace_training_curve"

    def __init__(
        self,
        train_logs,
        *args,
        energy_metric: str = "rmse_e_per_atom",
        force_metric: str = "rmse_f",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._train_logs = [str(p) for p in train_logs]
        self._energy_metric = energy_metric
        self._force_metric = force_metric

    def _read_eval_rows(self, path):
        """Yield the parsed ``mode == "eval"`` JSON rows of one log."""
        import json

        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("mode") == "eval":
                    yield rec

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        epochs, e_err, f_err = [], [], []
        for path in self._train_logs:
            for rec in self._read_eval_rows(path):
                epoch = rec.get("epoch")
                if epoch is None:  # the pre-training baseline eval
                    continue
                epochs.append(int(epoch))
                ev = rec.get(self._energy_metric)
                fv = rec.get(self._force_metric)
                e_err.append(float(ev) if ev is not None else float("nan"))
                f_err.append(float(fv) if fv is not None else float("nan"))
        return JsonDict(
            step=list(range(len(epochs))),
            epoch=epochs,
            e_err=e_err,
            f_err=f_err,
            energy_metric=self._energy_metric,
            force_metric=self._force_metric,
        )

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def plot_data(self):
        if not self.data["epoch"]:
            self.ax.set_title("MACE training (no eval rows)")
            return
        epoch = self.data["epoch"]
        e_err = np.asarray(self.data["e_err"], dtype=float)
        self.ax.plot(epoch, e_err, color="tab:blue", marker=".")
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel(
            f"energy {self.data['energy_metric']}", color="tab:blue"
        )
        if np.isfinite(e_err).any() and np.nanmin(e_err) > 0:
            self.ax.set_yscale("log")
        self.ax.set_title("MACE validation error")
        self.ax.grid(alpha=0.3)

        f_err = np.asarray(self.data["f_err"], dtype=float)
        if np.isfinite(f_err).any():
            ax2 = self.ax.twinx()
            ax2.plot(epoch, f_err, color="tab:red", marker=".")
            ax2.set_ylabel(
                f"force {self.data['force_metric']}", color="tab:red"
            )
            if np.nanmin(f_err) > 0:
                ax2.set_yscale("log")


def _build_neuralil_parity(ctx, c):
    return (
        NeuralILParityPanel(npz_files=ctx.npz_files, **c)
        if ctx.npz_files
        else None
    )


class SamplingParityPanel(ModelParityPanel):
    """DFT reference vs the *sampling* model's re-scored potential energy.

    Identical machinery to :class:`ModelParityPanel` -- only the model differs:
    the model that **proposed** these configs (the previous iteration's trained
    model that drove the NS sampler), not the current iteration's freshly trained
    one. So this is the genuine *out-of-sample* parity ("how well did the
    proposing model predict the points it selected"), whereas ``model_parity`` is
    an in-sample fit.

    Crucially it re-scores the model's **potential energy** rather than reading
    the trajectory's ``ns_energy``: for an NPT / ``EnsembleBackend`` run
    ``ns_energy`` is the NS *enthalpy* ``H = U + P*V`` (dominated by the ``P*V``
    term, tens of eV across a pressure grid), which is *not* comparable to the
    DFT potential energy. A subclass only to carry a distinct ``name`` -- entries
    cache/lay out by name, so it must not collide with ``model_parity``.
    """

    name = "sampling_parity"


class ModelParitySpec(PanelSpec):
    """Re-scored trained-model parity for one labelled set.

    ``source`` names the :class:`RunContext` attribute holding the configs to
    re-score (the evaluated set on the AL page; the validation split in the
    training section). ``show_forces`` / ``max_neighbors`` tune the backend
    re-score; their defaults match :class:`ModelParityPanel`, so the AL page is
    unchanged.
    """

    type: str = "model_parity"
    source: str = "eval_files"
    show_forces: bool = True
    remote_eval: bool | None = None
    max_neighbors: int = 50
    batch_size: int | None = None
    max_neighbors_list: list | None = None
    max_neighbors_offset: int = 4


def _build_model_parity(spec, ctx, c):
    # Only when there is no NeuralIL .npz bundle (MACE) -- else neuralil_parity
    # already shows real predicted-vs-ref parity.
    files = getattr(ctx, spec.source, None)
    if ctx.model_file and files and not ctx.npz_files:
        backend = ctx.eos_backend
        remote_eval = (
            spec.remote_eval
            if spec.remote_eval is not None
            else ctx.remote_model_eval
        )
        return ModelParityPanel(
            config_files=files,
            model_file=ctx.model_file,
            backend_type=backend,
            energy_ref_key=ctx.energy_key,
            forces_ref_key=ctx.forces_key,
            show_forces=spec.show_forces,
            remote_eval=remote_eval,
            max_neighbors=spec.max_neighbors,
            batch_size=spec.batch_size,
            max_neighbors_list=spec.max_neighbors_list,
            max_neighbors_offset=spec.max_neighbors_offset,
            **c,
        )
    return None


class SamplingParitySpec(PanelSpec):
    """DFT vs the *sampling* model's re-scored potential energy.

    The out-of-sample AL parity: re-score the evaluated configs with the model
    that proposed them (``ctx.sampling_model_file`` -- the previous iteration's
    trained model), instead of reading the trajectory's ``ns_energy`` (which for
    an NPT/``EnsembleBackend`` run is the enthalpy ``U + P*V``, not the potential
    energy the DFT reference carries). ``source`` names the configs; the other
    fields mirror :class:`ModelParitySpec`.
    """

    type: str = "sampling_parity"
    source: str = "eval_files"
    show_forces: bool = True
    remote_eval: bool | None = None
    max_neighbors: int = 50
    batch_size: int | None = None
    max_neighbors_list: list | None = None
    max_neighbors_offset: int = 4


def _build_sampling_parity(spec, ctx, c):
    # Needs the proposing model (absent on iteration 0 -- the foundation seed is
    # not recorded in AL_progress.json) and the evaluated configs to score.
    files = getattr(ctx, spec.source, None)
    if ctx.sampling_model_file and files:
        remote_eval = (
            spec.remote_eval
            if spec.remote_eval is not None
            else ctx.remote_model_eval
        )
        return SamplingParityPanel(
            config_files=files,
            model_file=ctx.sampling_model_file,
            backend_type=ctx.eos_backend,
            energy_ref_key=ctx.energy_key,
            forces_ref_key=ctx.forces_key,
            show_forces=spec.show_forces,
            remote_eval=remote_eval,
            max_neighbors=spec.max_neighbors,
            batch_size=spec.batch_size,
            max_neighbors_list=spec.max_neighbors_list,
            max_neighbors_offset=spec.max_neighbors_offset,
            **c,
        )
    return None


def _build_neuralil_training_curve(ctx, c):
    return (
        NeuralILTrainingCurve(error_files=ctx.error_files, **c)
        if ctx.error_files
        else None
    )


def _build_mace_training_curve(ctx, c):
    return (
        MaceTrainingCurve(train_logs=ctx.mace_logs, **c)
        if ctx.mace_logs
        else None
    )


register_simple("neuralil_parity", _build_neuralil_parity)
register_panel("model_parity", ModelParitySpec, _build_model_parity)
register_panel("sampling_parity", SamplingParitySpec, _build_sampling_parity)
register_simple("neuralil_training_curve", _build_neuralil_training_curve)
register_simple("mace_training_curve", _build_mace_training_curve)
