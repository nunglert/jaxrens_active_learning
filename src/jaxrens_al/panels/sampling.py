"""Project panel: how the jaxrens-sampled energies evolve across iterations.

A nested-sampling Sampler produces candidate configs carrying ``ns_energy``
(and ``run_id``). This panel reads those across AL iterations and shows the
per-iteration distribution of sampled energies -- a quick read on whether the
sampler keeps exploring (broad/shifting distributions) or has collapsed.

Project-specific (it knows jaxrens' ``ns_energy`` key and remote_handler's
layout), so it lives here rather than in alviz. Subclasses ``alviz.Entry`` and
stays pure (ase + numpy).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Panel spec.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, register_simple

from jaxrens_al.run_layout import sampled_configs_per_iteration


class NSSampledEnergyPanel(Entry):
    """Per-iteration distribution of sampled ``ns_energy`` (box per iteration).

    Two construction modes:

    * Cross-iteration (pass ``calchome``): one box per AL iteration, read from
      the whole run -- the original summary view.
    * Single-iteration (pass ``config_files`` + ``iteration_index``): a single
      box for one iteration's sampled configs -- used by the per-iteration
      multipage dashboard.

    Args:
        calchome: Path to a remote_handler run dir (cross-iteration mode).
        config_files: This iteration's sampled-config files (single-iteration
            mode). Mutually exclusive with ``calchome``.
        iteration_index: x-position/label of the single box in
            single-iteration mode.
        energy_key: Per-config energy key on the sampled configs
            (jaxrens writes ``ns_energy``).
        per_atom: Divide each config's energy by its atom count.
    """

    name = "sampled_ns_energy"

    def __init__(
        self,
        calchome=None,
        *args,
        config_files=None,
        iteration_index: int = 0,
        energy_key: str = "ns_energy",
        per_atom: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (calchome is None) == (config_files is None):
            raise ValueError(
                "NSSampledEnergyPanel: pass exactly one of 'calchome' "
                "(cross-iteration) or 'config_files' (single-iteration)."
            )
        self._calchome = calchome
        self._config_files = (
            [str(p) for p in config_files]
            if config_files is not None
            else None
        )
        self._iteration_index = iteration_index
        self._energy_key = energy_key
        self._per_atom = per_atom

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def _energies_from(self, paths) -> list:
        import ase.io

        energies = []
        for path in paths:
            for atoms in ase.io.read(path, index=":"):
                e = atoms.info.get(self._energy_key)
                if e is None:
                    continue
                norm = len(atoms) if self._per_atom else 1
                energies.append(float(e) / norm)
        return energies

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        if self._config_files is not None:
            groups = [(self._iteration_index, self._config_files)]
        else:
            groups = list(
                enumerate(sampled_configs_per_iteration(self._calchome))
            )

        iterations = []
        per_iter_energies = []
        for i, paths in groups:
            energies = self._energies_from(paths)
            if energies:
                iterations.append(i)
                per_iter_energies.append(energies)

        return JsonDict(
            iterations=iterations,
            energies=per_iter_energies,
            per_atom=self._per_atom,
        )

    def plot_data(self):
        iterations = self.data["iterations"]
        energies = self.data["energies"]
        unit = "eV/atom" if self.data["per_atom"] else "eV"

        if iterations:
            self.ax.boxplot(
                [np.asarray(e, dtype=float) for e in energies],
                positions=iterations,
                widths=0.6,
                showfliers=False,
            )
            # Median trend on top of the boxes.
            medians = [float(np.median(e)) for e in energies]
            self.ax.plot(
                iterations, medians, color="tab:red", marker="o", lw=1
            )

        self.ax.set_xlabel("AL iteration")
        self.ax.set_ylabel(f"sampled $E$ ({unit})")
        self.ax.set_title("Sampled-energy distribution")


def _build_sampled_ns_energy(ctx, c):
    if not ctx.sampled_files:
        return None
    if ctx.is_summary:
        return NSSampledEnergyPanel(calchome=ctx.calchome, **c)
    return NSSampledEnergyPanel(
        config_files=ctx.sampled_files, iteration_index=ctx.iteration, **c
    )


register_simple("sampled_ns_energy", _build_sampled_ns_energy)
