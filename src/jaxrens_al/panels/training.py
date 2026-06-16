"""Project panels for the NeuralIL training stage.

Two panels read what the NeuralIL trainer leaves on disk per iteration:

* :class:`NeuralILParityPanel` -- equips alviz's abstract ``ParityPanel``
  (rendering + ``ParityData`` contract) with computation: it reads the
  ``save_data`` ``.npz`` bundles (``e_obs``/``e_pred``/``f_obs``/``f_pred``)
  the trainer writes, instead of re-scoring a model. alviz owns the plot; we
  own *where the numbers come from*.
* :class:`NeuralILTrainingCurve` -- a from-scratch ``alviz.Entry`` (alviz has
  no training-curve skeleton): parses the per-epoch validation RMSE out of the
  trainer's ``error_*.txt`` log and plots E/F RMSE vs epoch.
"""

import re

import numpy as np
from alviz import Entry, JsonDict
from alviz.panels import ParityData, ParityPanel


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
