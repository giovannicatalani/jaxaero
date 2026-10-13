"""CGNS grid files (HDF5 storage), structured zones, read with h5py.

Per zone: the node coordinates (of a 3D zone one cell thick, the plane of
its thin index, as for Plot3D), the boundary faces of every ``BC_t`` given by
a ``PointRange``, named by the BC's family, else by its type (``BCWall*``:
``wall``, ``BCFarfield``: ``farfield``, ...), else by the node's name; and the
1-to-1 interfaces, which ``assemble`` checks its face matching against.
Files in the older ADF storage are refused with the conversion to use.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
ADF_SIGNATURE = b"ADF Database"

BC_TYPE_PATCHES = {
    "BCWall": "wall", "BCWallViscous": "wall", "BCWallViscousHeatFlux": "wall",
    "BCWallViscousIsothermal": "wall", "BCWallInviscid": "wall",
    "BCFarfield": "farfield", "BCSymmetryPlane": "symmetry",
    "BCInflow": "inflow", "BCInflowSubsonic": "inflow", "BCTunnelInflow": "inflow",
    "BCOutflow": "outflow", "BCOutflowSubsonic": "outflow", "BCTunnelOutflow": "outflow",
    "BCExtrapolate": "outflow",
}
"""Patch names for the standard CGNS boundary types, when a BC has no family."""


@dataclass(frozen=True)
class Declared:
    """A face range a file declares: a boundary ``name``, or (``name`` None)
    a 1-to-1 interface to zone ``donor``. Cells ``[start, stop)`` along
    ``side`` of the 2D block, in the file's own index order."""

    side: str
    start: int
    stop: int
    name: str | None = None
    donor: int | None = None


@dataclass(frozen=True)
class RawBlock:
    """A block as read: node coordinates ``(n_i, n_j)`` and what the file declares."""

    x: np.ndarray
    y: np.ndarray
    declared: tuple[Declared, ...] = ()
    name: str = ""


def _bytes(path: Path) -> bytes:
    data = Path(path).read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def is_cgns(path: Path) -> bool:
    head = _bytes(path)[:64]
    return head.startswith(HDF5_SIGNATURE) or ADF_SIGNATURE in head


def _label(node) -> str:
    value = node.attrs.get("label", b"")
    return (value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)).strip("\x00 ")


def _data(node):
    return node[" data"][()] if " data" in node else None


def _text(node) -> str:
    d = _data(node)
    return "" if d is None else d.tobytes().decode(errors="replace").strip("\x00 ")


def _children(node, label: str):
    return [(k, node[k]) for k in node if not k.startswith(" ") and _label(node[k]) == label]


def read_cgns(path: str | Path) -> list[RawBlock]:
    """Every structured zone of the first base, as a 2D block.

    Raises:
        ValueError: ADF storage, an unstructured zone, a BC without a
            ``PointRange``, or a 3D zone that is not a thin extrusion.
    """
    import h5py

    path = Path(path)
    data = _bytes(path)
    if ADF_SIGNATURE in data[:64]:
        raise ValueError(f"{path.name} is CGNS in the old ADF storage; convert it to HDF5 with the CGNS "
                         f"tools: adf2hdf {path.name} out.cgns (or cgnsconvert -h)")
    with h5py.File(io.BytesIO(data), "r") as f:
        bases = _children(f, "CGNSBase_t")
        if not bases:
            raise ValueError(f"{path.name}: no CGNSBase_t")
        zones = _children(bases[0][1], "Zone_t")
        names = [name for name, _ in zones]
        return [_zone(name, zone, names, path) for name, zone in zones]


def _zone(name: str, zone, zone_names: list[str], path: Path) -> RawBlock:
    types = _children(zone, "ZoneType_t")
    if types and _text(types[0][1]) != "Structured":
        raise ValueError(f"{path.name}: zone {name!r} is {_text(types[0][1])}; only structured zones are read")
    size = np.atleast_2d(_data(zone))[0]                      # vertex counts per index
    (_, coords), = _children(zone, "GridCoordinates_t")[:1]
    xyz = [np.asarray(_data(coords[f"Coordinate{c}"]), dtype=np.float64).T if f"Coordinate{c}" in coords
           else None for c in "XYZ"]
    if len(size) == 2:
        x, y, axes = xyz[0], xyz[1], (0, 1)
    else:
        x, y, axes = _thin_plane(xyz, name, path)
    declared = []
    for _, zbc in _children(zone, "ZoneBC_t"):
        for bc_name, bc in _children(zbc, "BC_t"):
            ranges = _children(bc, "IndexRange_t")
            if not ranges:
                raise ValueError(f"{path.name}: BC {bc_name!r} of zone {name!r} has no PointRange")
            face = _face(np.asarray(_data(ranges[0][1])), axes, size)
            if face is None:
                continue                                       # a face across the thin index
            family = _children(bc, "FamilyName_t")
            patch = (_text(family[0][1]) if family else None) or BC_TYPE_PATCHES.get(_text(bc)) or bc_name.strip()
            declared.append(Declared(*face, name=patch))
    for _, zgc in _children(zone, "ZoneGridConnectivity_t"):
        for _, conn in _children(zgc, "GridConnectivity1to1_t"):
            ranges = dict(_children(conn, "IndexRange_t"))
            face = _face(np.asarray(_data(ranges["PointRange"])), axes, size)
            if face is not None:
                donor = _text(conn)
                declared.append(Declared(*face, donor=zone_names.index(donor) if donor in zone_names else None))
    return RawBlock(x, y, tuple(declared), name)


def _thin_plane(xyz, name, path):
    """The plane of a 3D zone's thin index, in its two in-plane coordinates."""
    coords = [c for c in xyz if c is not None]
    scale = max(float(np.abs(c).max()) for c in coords) or 1.0
    for axis in range(3):
        if coords[0].shape[axis] > 2:
            continue
        first = [np.take(c, 0, axis) for c in xyz]
        last = [np.take(c, -1, axis) for c in xyz]
        keep = [k for k in range(3) if xyz[k] is not None
                and np.abs(first[k] - last[k]).max() <= 1e-10 * scale and np.ptp(first[k]) > 1e-10 * scale]
        if len(keep) == 2:
            return first[keep[0]], first[keep[1]], tuple(a for a in range(3) if a != axis)
    raise ValueError(f"{path.name}: zone {name!r} is 3D and not a thin extrusion of a 2D plane")


def _face(point_range: np.ndarray, axes: tuple[int, int], size) -> tuple[str, int, int] | None:
    """``(side, start, stop)`` cells of a node range ``(2, ndim)`` (begin,
    end; 1-based) in the 2D block of index axes ``axes``; None for a face
    across the dropped index."""
    begin, end = point_range[0], point_range[1]
    a, b = axes
    for constant, along, sides in ((a, b, ("i_min", "i_max")), (b, a, ("j_min", "j_max"))):
        if begin[constant] == end[constant] and begin[along] != end[along]:
            side = sides[0] if begin[constant] == 1 else sides[1]
            lo, hi = sorted((int(begin[along]), int(end[along])))
            return side, lo - 1, hi - 1
    return None
