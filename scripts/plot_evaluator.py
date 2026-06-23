"""Quick-look plots for an evaluator's labelled extxyz output.

Surfaces the things you usually want to eyeball before trusting an AL
iteration's DFT labels: energy/atom and force distributions, plus the
closest-pair distance per config (sub-Angstrom values flag degenerate
NS geometries that produce nonsensical REF_* labels).

Usage:
    python scripts/plot_evaluator.py <evaluator_it*.extxyz> [-o out.png] [--e0 -3.82]
    python scripts/plot_evaluator.py calchome/phase1_mace/0/evaluator_output/evaluator_it0_0.extxyz
"""

import argparse
from pathlib import Path

import ase.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect(ats):
    n = np.array([len(a) for a in ats])
    e = np.array([a.info["REF_energy"] for a in ats])
    epa = e / n
    vol = np.array([a.get_volume() for a in ats]) / n

    fmax, frms, dmin = [], [], []
    for a in ats:
        F = a.arrays["REF_forces"]
        mag = np.linalg.norm(F, axis=1)
        fmax.append(mag.max())
        frms.append(np.sqrt((mag**2).mean()))
        d = a.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        dmin.append(d.min())

    return dict(
        n=n,
        e=e,
        epa=epa,
        vol=vol,
        fmax=np.array(fmax),
        frms=np.array(frms),
        dmin=np.array(dmin),
        run_id=np.array([a.info.get("run_id", -1) for a in ats]),
        it=np.array([a.info.get("iter", -1) for a in ats]),
    )


def print_summary(d, src):
    print(f"Loaded {len(d['n'])} configs from {src}")
    print(f"  N_atoms:       {d['n'].min()}..{d['n'].max()}")
    print(f"  E_total:       [{d['e'].min():.2f}, {d['e'].max():.2f}] eV")
    print(
        f"  E/atom:        [{d['epa'].min():.3f}, {d['epa'].max():.3f}] eV/atom"
    )
    print(
        f"  |F|_max:       [{d['fmax'].min():.3f}, {d['fmax'].max():.3f}] eV/A   (mean {d['fmax'].mean():.2f})"
    )
    print(
        f"  |F|_rms:       [{d['frms'].min():.3f}, {d['frms'].max():.3f}] eV/A   (mean {d['frms'].mean():.2f})"
    )
    print(
        f"  min_dist:      [{d['dmin'].min():.3f}, {d['dmin'].max():.3f}] A   (mean {d['dmin'].mean():.3f})"
    )
    print(
        f"  V/atom:        [{d['vol'].min():.2f}, {d['vol'].max():.2f}] A^3/atom"
    )
    n_total = len(d["dmin"])
    print(
        f"  Configs with min_dist < 1.5 A: {(d['dmin'] < 1.5).sum()}/{n_total}"
    )
    print(
        f"  Configs with min_dist < 1.0 A: {(d['dmin'] < 1.0).sum()}/{n_total}"
    )


def plot_overview(d, out, title, e0=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.hist(d["epa"], bins=30, edgecolor="k")
    ax.set_xlabel("REF_energy / atom  [eV/atom]")
    ax.set_ylabel("count")
    ax.set_title("Energy/atom distribution")
    if e0 is not None:
        ax.axvline(e0, color="r", ls="--", lw=1, label=f"E0 = {e0:.2f}")
        ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(d["fmax"], bins=30, edgecolor="k", color="tab:orange")
    ax.set_xlabel("max |F|  [eV/A]")
    ax.set_ylabel("count")
    ax.set_title("Per-config max force magnitude")
    ax.set_yscale("log")

    ax = axes[0, 2]
    ax.hist(d["dmin"], bins=30, edgecolor="k", color="tab:green")
    ax.axvline(1.5, color="r", ls="--", lw=1, label="1.5 A (bond scale)")
    ax.axvline(1.0, color="darkred", ls=":", lw=1, label="1.0 A (degenerate)")
    ax.set_xlabel("min interatomic distance  [A]")
    ax.set_ylabel("count")
    ax.set_title("Closest-pair distance per config")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(d["dmin"], d["epa"], c=d["fmax"], cmap="viridis", s=20)
    ax.set_xlabel("min dist  [A]")
    ax.set_ylabel("E/atom  [eV/atom]")
    ax.set_title("Energy vs min-distance (color = max |F|)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("max |F|  [eV/A]")

    ax = axes[1, 1]
    ax.hist(d["vol"], bins=30, edgecolor="k", color="tab:purple")
    ax.set_xlabel("V / atom  [A^3/atom]")
    ax.set_ylabel("count")
    ax.set_title("Volume / atom distribution")

    ax = axes[1, 2]
    order = np.argsort(d["epa"])
    ax.plot(np.arange(len(d["epa"])), d["epa"][order], "o-", ms=3)
    ax.set_xlabel("config rank (sorted by E/atom)")
    ax.set_ylabel("REF_energy / atom  [eV/atom]")
    ax.set_title("Energy ordering (rank plot)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_provenance(d, out):
    if not np.any(d["run_id"] >= 0):
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    for rid in sorted(np.unique(d["run_id"][d["run_id"] >= 0])):
        m = d["run_id"] == rid
        ax.scatter(
            d["it"][m], d["epa"][m], label=f"run {rid}", s=18, alpha=0.7
        )
    ax.set_xlabel("NS iter (within run)")
    ax.set_ylabel("E/atom  [eV/atom]")
    ax.set_title("REF_energy / atom vs jaxrens NS iter (by walker run_id)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("extxyz", type=Path, help="labelled extxyz file to inspect")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output PNG (default: <extxyz stem>_analysis.png next to input)",
    )
    p.add_argument(
        "--e0",
        type=float,
        default=None,
        help="per-element reference energy in eV/atom to mark on the histogram (e.g. -3.82 for Sn)",
    )
    args = p.parse_args()

    ats = ase.io.read(args.extxyz, index=":")
    if not ats:
        raise SystemExit(f"{args.extxyz}: no configs")

    d = collect(ats)
    print_summary(d, args.extxyz)

    out = args.out or args.extxyz.with_name(f"{args.extxyz.stem}_analysis.png")
    plot_overview(
        d,
        out,
        title=f"DFT-evaluated configs: {args.extxyz.name}  ({len(ats)} configs)",
        e0=args.e0,
    )
    print(f"\nWrote {out}")

    prov_out = out.with_name(
        f"{out.stem.replace('_analysis', '')}_provenance.png"
    )
    if plot_provenance(d, prov_out):
        print(f"Wrote {prov_out}")


if __name__ == "__main__":
    main()
