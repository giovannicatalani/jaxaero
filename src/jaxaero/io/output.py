"""Result output and provenance.

Every result directory is self-describing: the resolved configuration, the git
commit it was produced at, the library versions, the device, and the residual
history travel with the fields. Without that, a plot found three months later
cannot be traced to the code that made it.

This module and ``plotting.py`` are the only places in the package that
touch the filesystem. Plotting reads these files and never the solver, so
redrawing a figure never forces a re-run (``python -m jaxaero.plot``).
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def git_revision(repo: str | Path | None = None) -> dict[str, Any]:
    """``{"commit", "dirty"}``; outside a repository the commit comes from a
    ``GIT_COMMIT`` file (a copy synced without ``.git``), else ``"unknown"`` -- never an error."""
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (subprocess.SubprocessError, OSError):
        # A copy synced without .git carries the hash in a file.
        stamp = repo / "GIT_COMMIT"
        if stamp.is_file():
            return {"commit": stamp.read_text().strip(), "dirty": False, "source": "GIT_COMMIT"}
        return {"commit": "unknown", "dirty": False}


def environment_record() -> dict[str, Any]:
    """Versions and device, for reproducing a run later."""
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import jax

        record["jax"] = jax.__version__
        record["jax_backend"] = jax.default_backend()
        record["jax_devices"] = [str(d) for d in jax.devices()]
        record["jax_device_kind"] = jax.devices()[0].device_kind
        record["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    except Exception:  # pragma: no cover - provenance must never break a run
        record["jax"] = "unavailable"
    return record


@dataclass
class ResultWriter:
    """Writes one run's files into ``directory`` (created), named by ``case_name``."""

    directory: Path
    case_name: str = "case"

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, configs: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
        """The resolved configs, ``extra`` fields and provenance, as JSON; returns the path."""
        payload = {
            "config": {name: asdict(c) for name, c in configs.items()},
            "git": git_revision(),
            "environment": environment_record(),
        }
        if extra:
            payload["extra"] = _jsonable(extra)
        path = self.directory / f"{self.case_name}_metadata.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def write_residual_history(self, residual: np.ndarray) -> Path:
        """Save the residual history as two columns: iteration, residual."""
        residual = np.asarray(residual)
        iters = np.arange(1, residual.size + 1)
        path = self.directory / f"{self.case_name}_residual.dat"
        np.savetxt(
            path,
            np.column_stack([iters, residual]),
            header="iteration residual_l2",
            fmt=["%d", "%.12e"],
        )
        return path

    def write_lift_history(self, cl: np.ndarray, block_iters: int) -> Path:
        """Lift at the end of every block: iteration, CL."""
        cl = np.asarray(cl)
        path = self.directory / f"{self.case_name}_lift.dat"
        np.savetxt(path, np.column_stack([block_iters * np.arange(1, cl.size + 1), cl]),
                   header="iteration cl", fmt=["%d", "%.12e"])
        return path

    def write_surface(self, surface: dict[str, np.ndarray]) -> Path:
        """Wall distributions, one row per wall face:
        ``x_over_c y_over_c body cp [cf]``, body by body in loop order from each
        trailing edge (post/forces.py, ``atlas_surface``)."""
        names = {"x": "x_over_c", "y": "y_over_c", "z": "z_over_c"}
        path = self.directory / f"{self.case_name}_surface.dat"
        np.savetxt(path, np.column_stack([np.asarray(v) for v in surface.values()]),
                   header=" ".join(names.get(k, k) for k in surface), fmt="%.12e")
        return path

    def write_fields(self, U_blocks: list[np.ndarray], nodes: list[np.ndarray]) -> Path:
        """The solution: per block ``k``, ``U_k`` ``(nvar, *cells)`` and its
        nodes ``x_node_k``, ``y_node_k`` [, ``z_node_k``]; and ``n_blocks``."""
        arrays = {"n_blocks": np.asarray(len(U_blocks))}
        for k, (U, xyz) in enumerate(zip(U_blocks, nodes)):
            arrays[f"U_{k}"] = U
            for name, c in zip("xyz", xyz):
                arrays[f"{name}_node_{k}"] = c
        path = self.directory / f"{self.case_name}_fields.npz"
        np.savez_compressed(path, **arrays)
        return path

    def write_vts(self, nodes: np.ndarray, cell_data: dict[str, np.ndarray], block: int = 0) -> Path:
        """One 3D block for ParaView: a VTK XML structured grid, raw appended
        binary, nodes ``(3, ni+1, nj+1, nk+1)``, cell fields ``(*cells)``
        (scalars) or ``(3, *cells)`` (vectors)."""
        path = self.directory / f"{self.case_name}_block{block}.vts"
        n = nodes.shape[1:]
        extent = f"0 {n[0] - 1} 0 {n[1] - 1} 0 {n[2] - 1}"
        payloads, arrays, offset = [], [], 0

        def add(name, values, components):
            nonlocal offset
            raw = np.ascontiguousarray(values, dtype="<f8").tobytes()
            payloads.append(np.uint64(len(raw)).tobytes() + raw)
            arrays.append((name, components, offset))
            offset += 8 + len(raw)

        def flat(a):      # VTK runs i fastest
            return a.transpose(tuple(range(a.ndim))[::-1]).ravel() if a.ndim == 3 else (
                np.stack([c.transpose(2, 1, 0).ravel() for c in a], axis=1).ravel())

        add("Points", flat(nodes), 3)
        for name, a in cell_data.items():
            add(name, flat(np.asarray(a)), 3 if np.asarray(a).ndim == 4 else 1)
        tags = lambda group: "".join(  # noqa: E731
            f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{c}" format="appended" offset="{o}"/>'
            for name, c, o in group)
        head = (f'<?xml version="1.0"?>\n<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" '
                f'header_type="UInt64"><StructuredGrid WholeExtent="{extent}"><Piece Extent="{extent}">'
                f"<Points>{tags(arrays[:1])}</Points><CellData>{tags(arrays[1:])}</CellData>"
                f'</Piece></StructuredGrid><AppendedData encoding="raw">_')
        with open(path, "wb") as out:
            out.write(head.encode())
            for chunk in payloads:
                out.write(chunk)
            out.write(b"</AppendedData></VTKFile>\n")
        return path

def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars and arrays into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


