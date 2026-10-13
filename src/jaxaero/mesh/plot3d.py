"""Plot3D grid files: every common variant, read to 2D blocks of nodes
(``read_plot3d``) or to 3D ones (``read_plot3d_3d``).

Formatted (text) or unformatted (Fortran records, either byte order),
single- or multi-block (with or without the block count), 2D or 3D (of a 3D
block the first plane of the index whose planes coincide in ``x, y`` is
taken: a one-cell spanwise extrusion), with or without iblank, optionally
gzipped. The layout is deduced from the sizes, never trusted from the file
name; a file that matches no layout exactly is refused.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

Block = tuple[np.ndarray, np.ndarray]
"""Node coordinates ``x, y``, ``(n_i, n_j)`` each."""


def _raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def read_plot3d(path: str | Path) -> list[Block]:
    """Every block of a Plot3D file, as 2D node arrays.

    Raises:
        ValueError: if the file matches no supported layout.
    """
    path = Path(path)
    raw = _raw(path)
    blocks = _unformatted(raw) if _looks_binary(raw) else _formatted(raw)
    if blocks is None:
        raise ValueError(f"{path.name}: not a Plot3D grid this reader understands (formatted or "
                         f"unformatted, 2D or 3D, single or multi-block, with or without iblank)")
    return [_plane(x, y, z) for x, y, z in blocks]


def read_plot3d_3d(path: str | Path) -> list[np.ndarray]:
    """Every block of a 3D Plot3D file, as nodes ``(3, n_i, n_j, n_k)``.

    Raises:
        ValueError: if the file matches no supported layout or is 2D.
    """
    path = Path(path)
    raw = _raw(path)
    blocks = _unformatted(raw) if _looks_binary(raw) else _formatted(raw)
    if blocks is None:
        raise ValueError(f"{path.name}: not a Plot3D grid this reader understands")
    if any(z is None for _, _, z in blocks):
        raise ValueError(f"{path.name}: a 2D grid; read it with read_plot3d")
    return [np.stack([x, y, z]) for x, y, z in blocks]


def _looks_binary(raw: bytes) -> bool:
    head = raw[:4096]
    return any(b < 9 or 13 < b < 32 for b in head)


def _plane(x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> Block:
    """A 2D block: as given, or of a 3D one the first plane of a thin index
    (one or two planes) in the two coordinates that do not vary across it --
    ``x, y`` for a grid in the z = const plane, ``x, z`` for one extruded in
    ``y`` (CFL3D's convention)."""
    if x.ndim == 2:
        return np.ascontiguousarray(x), np.ascontiguousarray(y)
    coords = (x, y, z)
    scale = max(float(np.abs(c).max()) for c in coords) or 1.0
    for axis in range(3):
        if x.shape[axis] > 2:
            continue
        first = [np.take(c, 0, axis) for c in coords]
        last = [np.take(c, -1, axis) for c in coords]
        same = [np.abs(a - b).max() <= 1e-10 * scale for a, b in zip(first, last)]
        flat = [np.ptp(a) <= 1e-10 * scale for a in first]          # constant over the plane
        in_plane = [k for k in range(3) if same[k] and not flat[k]]
        if len(in_plane) == 2:
            return tuple(np.ascontiguousarray(first[k]) for k in in_plane)
    raise ValueError(f"a 3D block of {x.shape} nodes is not a 2D plane extruded along a thin index")


def _split(values: np.ndarray, dims: np.ndarray, ndim: int, iblank: bool):
    """Coordinates of each block from the flat values, first index fastest."""
    blocks, at = [], 0
    for shape in dims:
        n = int(np.prod(shape))
        coords = []
        for _ in range(ndim):
            coords.append(values[at:at + n].reshape(shape[::-1]).T)
            at += n
        if iblank:
            at += n
        blocks.append((coords[0], coords[1], coords[2] if ndim == 3 else None))
    return blocks if at == values.size else None


def _formatted(raw: bytes):
    text = raw.decode(errors="replace").replace("D", "E").replace("d", "e")
    numbers = np.fromstring(text, dtype=np.float64, sep=" ")   # C-level; ~1M tokens per second
    if numbers.size == 0 or numbers.size != len(text.split()):
        return None
    for multi in (True, False):
        for ndim in (2, 3):
            head = 1 if multi else 0
            if numbers.size <= head:
                continue
            nblocks = int(numbers[0]) if multi else 1
            if multi and (numbers[0] != nblocks or nblocks < 1):
                continue
            dims = numbers[head:head + ndim * nblocks]
            if dims.size < ndim * nblocks or np.any(dims != np.round(dims)) or np.any(dims < 1):
                continue
            dims = dims.astype(np.int64).reshape(nblocks, ndim)
            values = numbers[head + ndim * nblocks:]
            n = int(dims.prod(axis=1).sum())
            for iblank in (False, True):
                if values.size == n * (ndim + iblank):
                    if iblank:      # formatted iblank follows each block's coordinates
                        return _split(values, dims, ndim, True)
                    return _split(values, dims, ndim, False)
    return None


def _records(raw: bytes, order: str):
    """Fortran sequential records, or None if the markers do not hold."""
    out, at = [], 0
    while at < len(raw):
        if at + 4 > len(raw):
            return None
        n = int(np.frombuffer(raw, order + "i4", 1, at)[0])
        if n < 0 or at + 8 + n > len(raw) or int(np.frombuffer(raw, order + "i4", 1, at + 4 + n)[0]) != n:
            return None
        out.append(raw[at + 4:at + 4 + n])
        at += n + 8
    return out


def _unformatted(raw: bytes):
    for order in ("<", ">"):
        records = _records(raw, order)
        if not records:
            continue
        ints = lambda r: np.frombuffer(r, order + "i4")  # noqa: E731
        for multi in (True, False):
            k = 0
            if multi:
                if len(records[0]) != 4:
                    continue
                nblocks, k = int(ints(records[0])[0]), 1
            else:
                nblocks = 1
            if k >= len(records):
                continue
            dims = ints(records[k])
            for ndim in (2, 3):
                if dims.size != ndim * nblocks or np.any(dims < 1):
                    continue
                shapes = dims.reshape(nblocks, ndim)
                data = records[k + 1:]
                if len(data) != nblocks:
                    continue
                blocks = []
                for shape, record in zip(shapes, data):
                    n = int(np.prod(shape))
                    for precision in ("f8", "f4"):
                        size = np.dtype(precision).itemsize
                        iblank = len(record) == n * (ndim * size + 4)
                        if len(record) == n * ndim * size or iblank:
                            values = np.frombuffer(record, order + precision, n * ndim).astype(np.float64)
                            coords = [values[c * n:(c + 1) * n].reshape(shape[::-1]).T for c in range(ndim)]
                            blocks.append((coords[0], coords[1], coords[2] if ndim == 3 else None))
                            break
                    else:
                        break
                if len(blocks) == nblocks:
                    return blocks
    return None
