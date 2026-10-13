"""Figures from written results: fields, wall distributions, convergence, polars.

    python -m jaxaero.plot RESULT_DIR [--out DIR] [--window XMIN XMAX YMIN YMAX]
    python -m jaxaero.plot POLAR_DIR

Reads only what ``Result.write`` / ``Polar.write`` saved, never the solver, so
results computed on a cluster are plotted anywhere. Writes PNGs to
``<dir>/figures/``: ``mach.png``, ``cp_field.png``, ``cp.png``, ``cf.png``
(viscous), ``convergence.png``; for a polar ``polar.png`` and ``cp_family.png``.
Comparison curves are the caller's (``validation/report.py``); this module
knows no reference. Needs matplotlib (``pip install -e .[plot]``).

Colours: magnitudes (Mach) on a one-hue sequential ramp; Cp, whose sign is the
point, diverging about Cp = 0; line series in a fixed order -- this code,
reference code, experiment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MESH = "#a9a7a1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGING = ("#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a32c2c")
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2.0,
}
WINDOW = (-0.4, 1.4, -0.6, 0.6)

Curve = tuple[np.ndarray, np.ndarray]


# --- Reading ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResultFiles:
    """A directory written by ``Result.write``: per block ``(U, x_node, y_node)``,
    the wall ``surface`` columns by name (``x``, ``y``, ``body``, ``cp`` [, ``cf``]),
    histories."""

    name: str
    metadata: dict
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    surface: dict[str, np.ndarray]
    residual: np.ndarray
    cl_history: np.ndarray

    @property
    def flow(self) -> dict:
        return self.metadata["config"]["flow"]

    @property
    def extra(self) -> dict:
        return self.metadata.get("extra", {})

    def bodies(self) -> list[dict[str, np.ndarray]]:
        """The surface columns of each body, in loop order."""
        body = self.surface["body"].astype(int)
        return [{k: v[body == b] for k, v in self.surface.items()} for b in range(body.max() + 1)]


def read_table(path: Path) -> dict[str, np.ndarray]:
    """Columns of a whitespace table whose header line (``# a b c``) names them."""
    names = path.read_text().splitlines()[0].lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    short = {"x_over_c": "x", "y_over_c": "y"}
    return {short.get(n, n): data[:, k] for k, n in enumerate(names)}


def load_result(directory: str | Path) -> ResultFiles:
    directory = Path(directory)

    def one(suffix: str) -> Path:
        found = sorted(directory.glob(f"*{suffix}"))
        if len(found) != 1:
            raise FileNotFoundError(f"{directory}: expected one *{suffix}, found {len(found)}")
        return found[0]

    meta = one("_metadata.json")
    fields = np.load(one("_fields.npz"))
    if "z_node_0" in fields.files:
        raise SystemExit(f"{directory}: a 3D result; these figures are 2D only "
                         "(3D wings: scripts/plot_wing.py, or the .vts file in ParaView)")
    blocks = [(fields[f"U_{k}"], fields[f"x_node_{k}"], fields[f"y_node_{k}"])
              for k in range(int(fields["n_blocks"]))]
    return ResultFiles(
        name=meta.name.removesuffix("_metadata.json"), metadata=json.loads(meta.read_text()),
        blocks=blocks, surface=read_table(one("_surface.dat")),
        residual=np.loadtxt(one("_residual.dat"), ndmin=2)[:, 1],
        cl_history=np.loadtxt(one("_lift.dat"), ndmin=2)[:, 1],
    )


def load_polar(directory: str | Path) -> tuple[np.ndarray, dict]:
    """``polar.dat`` rows and ``polar_metadata.json``."""
    directory = Path(directory)
    return (np.loadtxt(directory / "polar.dat", ndmin=2),
            json.loads((directory / "polar_metadata.json").read_text()))


def mach_and_cp(U: np.ndarray, mach_inf: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per cell; in the code's units ``p_inf = 1/gamma``, ``q_inf = M^2/2``."""
    rho, u, v = U[0], U[1] / U[0], U[2] / U[0]
    p = (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v))
    return np.sqrt((u * u + v * v) * rho / (gamma * p)), (p - 1.0 / gamma) / (0.5 * mach_inf**2)


# --- Drawing ----------------------------------------------------------------------


def _figure(**kwargs):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("plotting needs matplotlib: pip install -e .[plot]") from error
    plt.rcParams.update(STYLE)
    return plt.subplots(layout="constrained", **kwargs)


def _save(fig, out: Path | None):
    if out is not None:
        import matplotlib.pyplot as plt

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def _ramp(colours):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ramp", colours)


def _title(res: ResultFiles, what: str) -> str:
    f = res.flow
    re = "" if f.get("reynolds") is None else f", Re {f['reynolds']:.3g}"
    return f"{what}: {res.name}  (M {f['mach']:g}, α {f['alpha_deg']:g}°{re})"


def plot_field(res: ResultFiles, which: str, window=WINDOW, out: Path | None = None):
    """Cell values of ``"mach"`` (sonic line drawn) or ``"cp"`` in ``window``,
    block by block, the bodies filled."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Polygon
    from matplotlib.ticker import MaxNLocator

    values = [mach_and_cp(U, res.flow["mach"], res.flow["gamma"])[0 if which == "mach" else 1]
              for U, _, _ in res.blocks]
    seen = np.concatenate([v[_in_window(x, y, window)] for v, (_, x, y) in zip(values, res.blocks)])
    if which == "mach":
        norm, cmap = Normalize(0.0, seen.max()), _ramp(SEQUENTIAL)
    else:
        lo, hi = min(seen.min(), -0.1), max(seen.max(), 0.1)
        norm, cmap = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi), _ramp(DIVERGING)
    fig, ax = _figure(figsize=(8.0, 4.6))
    for v, (_, x, y) in zip(values, res.blocks):
        mesh = ax.pcolormesh(x, y, v, norm=norm, cmap=cmap, shading="flat", rasterized=True)
        if which == "mach" and v.max() > 1.0:
            xc = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[1:, 1:] + x[:-1, 1:])
            yc = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[1:, 1:] + y[:-1, 1:])
            ax.contour(xc, yc, v, levels=[1.0], colors=TEXT, linewidths=1.2)
    for body in res.bodies():
        ax.add_patch(Polygon(np.column_stack([body["x"], body["y"]]), facecolor=TEXT_SECONDARY,
                             edgecolor=TEXT, linewidth=0.8, zorder=5))
    label = "Mach number" if which == "mach" else "pressure coefficient Cp"
    ax.set(xlim=window[:2], ylim=window[2:], aspect="equal", xlabel="x / c", ylabel="y / c",
           title=_title(res, label.split(" Cp")[0].capitalize()))
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label=label, shrink=0.85, ticks=MaxNLocator(8))
    return _save(fig, out)


def _in_window(x, y, window):
    xc = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[1:, 1:] + x[:-1, 1:])
    yc = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[1:, 1:] + y[:-1, 1:])
    return (xc > window[0]) & (xc < window[1]) & (yc > window[2]) & (yc < window[3])


def plot_surface(res: ResultFiles, quantity: str = "cp", references: dict[str, Curve] | None = None,
                 out: Path | None = None):
    """Wall ``cp`` (suction up) or ``cf`` against x/c, body by body, with
    ``references`` (label -> points; dense ones drawn as lines) in slots 2 and 3."""
    fig, ax = _figure(figsize=(7.0, 4.4))
    bodies = res.bodies()
    for b, body in enumerate(bodies):
        x, value = body["x"], body[quantity]
        name = "this code" + (f", body {b}" if len(bodies) > 1 else "")
        k = int(np.argmin(x))
        if quantity == "cf" and 0 < k < x.size - 1:      # an airfoil: upper and lower from the LE
            ax.plot(x[k:], value[k:], color=SERIES[0], label=f"{name}, upper" if b == 0 else None)
            ax.plot(x[: k + 1], value[: k + 1], color=SERIES[0], linestyle=(0, (4, 2)),
                    label=f"{name}, lower" if b == 0 else None)
        else:
            closed = np.append(value, value[:1]) if k > 0 else value
            ax.plot(np.append(x, x[:1]) if k > 0 else x, closed, color=SERIES[0],
                    label="this code" if b == 0 else None)
    if quantity == "cf":
        ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8)
    for slot, (label, (xr, yr)) in enumerate((references or {}).items(), start=1):
        style = (dict(linestyle=(0, (5, 2)), linewidth=1.5) if len(xr) > 120 else
                 dict(linestyle="none", marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8))
        ax.plot(xr, yr, color=SERIES[slot], label=label, **style)
    if quantity == "cp":
        ax.invert_yaxis()
    ax.set(xlabel="x / c", ylabel=quantity.capitalize(),
           title=_title(res, "Wall pressure" if quantity == "cp" else "Skin friction"))
    ax.legend(loc="best")
    return _save(fig, out)


def plot_grid(atlas, window=WINDOW, out: Path | None = None):
    """The grid near the bodies: block edges dark, walls coloured by body."""
    fig, axes = _figure(ncols=2, figsize=(12, 5))
    for ax, win in zip(axes, (None, window)):
        for block in atlas.mesh:
            x, y = np.asarray(block.x_node), np.asarray(block.y_node)
            ax.plot(x, y, color=MESH, lw=0.3)
            ax.plot(x.T, y.T, color=MESH, lw=0.3)
            for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
                ax.plot(x[edge], y[edge], color=TEXT_SECONDARY, lw=0.6)
        walls = atlas.walls.faces
        for body in range(walls.bodies):
            a, b = walls.a[walls.body == body], walls.b[walls.body == body]
            colour = SERIES[body % len(SERIES)]
            ax.plot(np.c_[a[:, 0], b[:, 0]].T, np.c_[a[:, 1], b[:, 1]].T, color=colour, lw=1.5)
            if win is not None:
                ax.annotate(f"body {body}", a[np.argmin(a[:, 0])], color=colour, fontsize=9,
                            xytext=(-30, 8), textcoords="offset points")
        ax.set_aspect("equal")
        ax.grid(False)
        if win is not None:
            ax.set(xlim=win[:2], ylim=win[2:])
    axes[0].set_title("whole domain")
    axes[1].set_title("near the bodies")
    return _save(fig, out)


def plot_convergence(res: ResultFiles, out: Path | None = None):
    fig, ax = _figure(figsize=(7.0, 3.6))
    ax.semilogy(np.arange(1, res.residual.size + 1), res.residual, color=SERIES[0], linewidth=1.5)
    ax.set(xlabel="iteration", ylabel="density residual (L2)", title=f"Convergence: {res.name}")
    return _save(fig, out)


def plot_result(directory: str | Path, out: str | Path | None = None, window=WINDOW,
                references: dict[str, dict[str, Curve]] | None = None) -> list[Path]:
    """Every standard figure of one result, with ``references`` per quantity
    (``"cp"``, ``"cf"``); returns the written paths."""
    res = load_result(directory)
    out = Path(out) if out is not None else Path(directory) / "figures"
    references = references or {}
    paths = {name: out / f"{name}.png" for name in ("mach", "cp_field", "cp", "cf", "convergence")}
    plot_field(res, "mach", window, paths["mach"])
    plot_field(res, "cp", window, paths["cp_field"])
    plot_surface(res, "cp", references.get("cp"), paths["cp"])
    if "cf" in res.surface:
        plot_surface(res, "cf", references.get("cf"), paths["cf"])
    else:
        del paths["cf"]
    plot_convergence(res, paths["convergence"])
    return list(paths.values())


def plot_polar(directory: str | Path, out: Path | None = None):
    """CL-, CD-, CM-alpha and the drag polar; angles whose CL did not settle hollow."""
    table, meta = load_polar(directory)
    alpha, cl, cd, _, cm = table[:, :5].T
    steady = table[:, -1] > 0.5
    fig, axes = _figure(nrows=2, ncols=2, figsize=(9.0, 7.0))
    for ax, x, y, labels in ((axes[0, 0], alpha, cl, ("α (deg)", "CL")), (axes[0, 1], alpha, cd, ("α (deg)", "CD")),
                             (axes[1, 0], alpha, cm, ("α (deg)", "CM (c/4, nose-up +)")),
                             (axes[1, 1], cd, cl, ("CD", "CL"))):
        ax.plot(x, y, color=SERIES[0], linewidth=1.5)
        ax.plot(x[steady], y[steady], "o", color=SERIES[0], markeredgecolor=SURFACE, label="steady (CL settled)")
        ax.plot(x[~steady], y[~steady], "o", markerfacecolor=SURFACE, markeredgecolor=SERIES[0],
                markeredgewidth=1.5, label="not settled")
        ax.set(xlabel=labels[0], ylabel=labels[1])
    axes[0, 0].legend(loc="upper left")
    flow = meta["case_setup"]["case"]["freestream"]
    fig.suptitle(f"Polar: {meta['case']}  (M {flow['mach']:g}, Re {flow.get('reynolds') or 0:.3g})", fontweight="bold")
    return _save(fig, out)


def plot_cp_family(directory: str | Path, alphas=None, out: Path | None = None):
    """Wall Cp at several settled angles of a polar, angle on a sequential ramp."""
    table, meta = load_polar(directory)
    settled = table[table[:, -1] > 0.5, 0]
    alphas = settled[:: max(1, settled.size // 6)] if alphas is None else alphas
    ramp = _ramp(SEQUENTIAL[2:])          # from step 250: every shade readable on the surface
    fig, ax = _figure(figsize=(7.5, 4.8))
    for k, alpha in enumerate(alphas):
        surface = load_result(Path(directory) / f"alpha_{alpha:+.2f}").surface
        ax.plot(surface["x"], surface["cp"], color=ramp(k / max(1, len(alphas) - 1)), linewidth=1.6,
                label=f"α = {alpha:+.1f}°")
    ax.invert_yaxis()
    ax.set(xlabel="x / c", ylabel="Cp", title=f"Wall pressure across the polar: {meta['case']}")
    ax.legend(loc="lower right", ncols=2)
    return _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jaxaero.plot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="a result or polar directory")
    parser.add_argument("--out", type=Path, help="figure directory (default: DIRECTORY/figures)")
    parser.add_argument("--window", type=float, nargs=4, default=WINDOW,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="field plot window, chords")
    args = parser.parse_args(argv)
    out = args.out or args.directory / "figures"
    if (args.directory / "polar.dat").is_file():
        plot_polar(args.directory, out / "polar.png")
        plot_cp_family(args.directory, out=out / "cp_family.png")
        paths = [out / "polar.png", out / "cp_family.png"]
    else:
        paths = plot_result(args.directory, out, tuple(args.window))
    print("\n".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
