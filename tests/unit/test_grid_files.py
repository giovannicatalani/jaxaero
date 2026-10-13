"""Grid files: every Plot3D variant and CGNS read to the same blocks, and the
one assembly (interfaces, patches) whatever the source.

Synthetic files are written here, so these tests need nothing on disk; the
TMR CGNS/Plot3D comparison skips without data/grids.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from jaxaero.mesh.assemble import PatchRule, RawBlock, assemble, load_grid
from jaxaero.mesh.block import Interface, o_grid_sides
from jaxaero.mesh.cgns import Declared
from validation.grids.karman_trefftz import make_o_grid
from jaxaero.mesh.plot3d import read_plot3d

GRIDS = Path(__file__).resolve().parents[2] / "data" / "grids"


def block(ni=9, nj=5, seed=0):
    """An annulus sector: nodes ``(ni, nj)``, right-handed."""
    r = np.linspace(1.0, 2.0, nj)[None, :]
    t = np.linspace(0.0, 1.5, ni)[:, None]
    return r * np.cos(t), r * np.sin(t)


# --- Plot3D -------------------------------------------------------------------------


def write_formatted(path, blocks, multi=True, ndim=2, iblank=False):
    lines = [str(len(blocks))] if multi else []
    for x, _ in blocks:
        lines.append(" ".join(map(str, x.shape + ((1,) if ndim == 3 else ()))))
    for x, y in blocks:
        coords = [x, y] + ([np.zeros_like(x)] if ndim == 3 else [])
        for c in coords:
            lines.append(" ".join(f"{v:.17e}" for v in c.T.ravel()))
        if iblank:
            lines.append(" ".join("1" for _ in x.ravel()))
    Path(path).write_text("\n".join(lines) + "\n")


def record(data: bytes, order: str) -> bytes:
    n = np.array([len(data)], dtype=order + "i4").tobytes()
    return n + data + n


def write_unformatted(path, blocks, order="<", multi=True, ndim=2, precision="f8", iblank=False):
    out = b""
    if multi:
        out += record(np.array([len(blocks)], order + "i4").tobytes(), order)
    dims = [x.shape + ((1,) if ndim == 3 else ()) for x, _ in blocks]
    out += record(np.array(dims, order + "i4").ravel().tobytes(), order)
    for x, y in blocks:
        coords = [x, y] + ([np.zeros_like(x)] if ndim == 3 else [])
        data = b"".join(np.ascontiguousarray(c.T).astype(order + precision).tobytes() for c in coords)
        if iblank:
            data += np.ones(x.size, order + "i4").tobytes()
        out += record(data, order)
    Path(path).write_bytes(out)


@pytest.mark.parametrize("variant", [
    dict(kind="formatted", multi=True, ndim=2), dict(kind="formatted", multi=False, ndim=2),
    dict(kind="formatted", multi=True, ndim=3), dict(kind="formatted", multi=True, ndim=2, iblank=True),
    dict(kind="unformatted", order="<", multi=True, ndim=2), dict(kind="unformatted", order=">", multi=True, ndim=3),
    dict(kind="unformatted", order="<", multi=False, ndim=2), dict(kind="unformatted", order="<", multi=True, ndim=3, iblank=True),
    dict(kind="unformatted", order="<", multi=True, ndim=2, precision="f4"),
])
@pytest.mark.parametrize("gz", [False, True])
def test_every_plot3d_variant_reads_back(tmp_path, variant, gz) -> None:
    variant = dict(variant)
    blocks = [block(9, 5), block(6, 4)] if variant.get("multi", True) else [block(9, 5)]
    path = tmp_path / "grid.xyz"
    kind = variant.pop("kind")
    (write_formatted if kind == "formatted" else write_unformatted)(path, blocks, **variant)
    if gz:
        path.write_bytes(gzip.compress(path.read_bytes()))
    got = read_plot3d(path)
    tol = 1e-6 if variant.get("precision") == "f4" else 1e-15
    assert len(got) == len(blocks)
    for (x, y), (gx, gy) in zip(blocks, got):
        assert np.allclose(gx, x, atol=tol) and np.allclose(gy, y, atol=tol)


def test_a_grid_extruded_in_y_is_read_in_its_x_z_plane(tmp_path) -> None:
    """CFL3D's convention (TMR's .p3dfmt): span along y, airfoil in x-z."""
    x, z = block()
    path = tmp_path / "grid.p3dfmt"
    lines = ["1", f"2 {x.shape[0]} {x.shape[1]}"]
    for c in (np.stack([x, x]), np.stack([np.zeros_like(x), np.ones_like(x)]), np.stack([z, z])):
        lines.append(" ".join(f"{v:.17e}" for v in np.transpose(c, (2, 1, 0)).ravel()))
    path.write_text("\n".join(lines))
    (gx, gy), = read_plot3d(path)
    assert np.array_equal(gx, x) and np.array_equal(gy, z)


def test_a_file_matching_no_layout_is_refused(tmp_path) -> None:
    path = tmp_path / "grid.xyz"
    path.write_text("1\n3 3\n0 1 2\n")
    with pytest.raises(ValueError, match="not a Plot3D grid"):
        read_plot3d(path)


# --- Assembly ------------------------------------------------------------------------


def o_nodes(n=16):
    grid = make_o_grid(n)
    return np.asarray(grid.x_node), np.asarray(grid.y_node)


def test_an_o_grid_file_gets_its_seam_wall_and_far_field() -> None:
    mesh, info = assemble([RawBlock(*o_nodes())])
    assert mesh[0].sides == o_grid_sides(16, 16)
    assert info.patch_faces == {"wall": 16, "farfield": 16} and info.bodies == 1


def test_split_blocks_are_reassembled_with_interfaces() -> None:
    """Two halves of an O-grid: two interfaces each way, same patches."""
    x, y = o_nodes()
    mesh, info = assemble([RawBlock(x[:9], y[:9]), RawBlock(x[8:], y[8:])])
    assert info.interfaces == 4
    for block_, other in ((0, 1), (1, 0)):
        for side in ("i_min", "i_max"):
            (seg,) = mesh[block_].sides.side(side)
            assert isinstance(seg, Interface) and seg.block == other


def test_patch_rules_name_faces_by_box_and_the_file_overrides_them() -> None:
    x, y = o_nodes()
    rules = (PatchRule("upper", y_min=0.0, x_max=0.9), PatchRule("outer", x_min=-1e9))
    mesh, info = assemble([RawBlock(x, y)], rules)
    assert set(info.patch_faces) == {"upper", "outer"}
    declared = (Declared("j_min", 0, 16, name="airfoil"),)
    mesh, info = assemble([RawBlock(x, y, declared)], rules)
    assert info.patch_faces["airfoil"] == 16


def test_a_declared_boundary_is_never_matched() -> None:
    """A zero-thickness plate: two coincident walls stay walls, not a cut."""
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(-1.0, 0.0, 4), indexing="ij")
    below = RawBlock(x, y, (Declared("j_max", 1, 3, name="plate"),))
    above = RawBlock(x, y + 1.0, (Declared("j_min", 1, 3, name="plate"),))
    mesh, info = assemble([below, above])
    assert info.patch_faces["plate"] == 4
    assert [type(s).__name__ for s in mesh[0].sides.j_max] == ["Interface", "Boundary", "Interface"]


def test_boundary_loops_name_the_outer_one_far_field_and_bodies_wall() -> None:
    """A rectangle with a hole: the rectangle is far field, the hole a wall,
    wherever the hole sits in the box."""
    def ring(n, cx):
        t = np.linspace(0.0, 2 * np.pi, n + 1)[::-1]          # clockwise: right-handed with j outward
        r = np.linspace(0.2, 3.0, 6)[None, :]
        return cx + 0.0 + r * np.cos(t)[:, None], r * np.sin(t)[:, None]

    for cx in (0.0, 2.5):
        mesh, info = assemble([RawBlock(*ring(24, cx))])
        assert [s.patch for s in mesh[0].sides.j_min] == ["wall"]
        assert [s.patch for s in mesh[0].sides.j_max] == ["farfield"]


# --- CGNS ----------------------------------------------------------------------------


def write_cgns(path, zones, unstructured=False):
    """A minimal CGNS/HDF5 file: per zone ``(x, y[, z], bcs)``, ``bcs`` a list
    of ``(name, type, family, begin, end)`` (1-based node index ranges)."""
    h5py = pytest.importorskip("h5py")

    def node(parent, name, label, data=None):
        g = parent.create_group(name)
        g.attrs["label"] = np.bytes_(label)
        if data is not None:
            g.create_dataset(" data", data=data)
        return g

    def text(s):
        return np.frombuffer(s.encode(), dtype=np.int8)

    with h5py.File(path, "w") as f:
        base = node(f, "Base", "CGNSBase_t", np.array([3 if zones[0][0][0].ndim == 3 else 2, 3], np.int32))
        for k, (coords, bcs) in enumerate(zones):
            shape = coords[0].shape
            size = np.array([list(shape), [n - 1 for n in shape], [0] * len(shape)], np.int32)
            z = node(base, f"zone{k}", "Zone_t", size)
            node(z, "ZoneType", "ZoneType_t", text("Unstructured" if unstructured else "Structured"))
            gc = node(z, "GridCoordinates", "GridCoordinates_t")
            for c, name in zip(coords, ("CoordinateX", "CoordinateY", "CoordinateZ")):
                node(gc, name, "DataArray_t", np.ascontiguousarray(c.T))
            zbc = node(z, "ZoneBC", "ZoneBC_t")
            for name, bctype, family, begin, end in bcs:
                bc = node(zbc, name, "BC_t", text(bctype))
                node(bc, "PointRange", "IndexRange_t", np.array([begin, end], np.int32))
                if family:
                    node(bc, "FamilyName", "FamilyName_t", text(family))


def test_cgns_names_patches_by_family_then_type_then_node(tmp_path) -> None:
    x, y = o_nodes()
    ni, nj = x.shape
    bcs = [("Seg1", "BCWallViscous", None, [1, 1], [ni, 1]),
           ("Seg2", "BCFarfield", "outer", [1, nj], [ni, nj])]
    path = tmp_path / "o.cgns"
    write_cgns(path, [((x, y), bcs)])
    mesh, info = load_grid(path)
    assert info.patch_faces == {"wall": 16, "outer": 16}
    assert mesh[0].sides.i_min[0].block == 0                  # the seam, by matching


def test_a_thin_3d_cgns_zone_reads_as_its_plane(tmp_path) -> None:
    """TMR's layout: span index first (2 planes, along y), the plane in x-z;
    BCs across the span index are dropped."""
    x, z = o_nodes()
    ni, nj = x.shape
    X = np.stack([x, x]); Y = np.stack([np.zeros_like(x), np.ones_like(x)]); Z = np.stack([z, z])
    bcs = [("span", "BCSymmetryPlane", None, [1, 1, 1], [1, ni, nj]),
           ("wall", "BCWallViscousHeatFlux", None, [1, 1, 1], [2, ni, 1])]
    path = tmp_path / "o3.cgns"
    write_cgns(path, [((X, Y, Z), bcs)])
    mesh, info = load_grid(path)
    assert np.array_equal(np.asarray(mesh[0].x_node), x) and np.array_equal(np.asarray(mesh[0].y_node), z)
    assert info.patch_faces == {"wall": 16, "farfield": 16}


def test_unstructured_and_adf_cgns_are_refused_clearly(tmp_path) -> None:
    x, y = o_nodes()
    path = tmp_path / "u.cgns"
    write_cgns(path, [((x, y), [])], unstructured=True)
    with pytest.raises(ValueError, match="only structured zones"):
        load_grid(path)
    adf = tmp_path / "a.cgns"
    adf.write_bytes(b"\xc0\xa8\xa3\xa9ADF Database Version A02011>" + b"\0" * 64)
    with pytest.raises(ValueError, match="adf2hdf"):
        load_grid(adf)


def test_tmr_cgns_and_plot3d_give_one_mesh() -> None:
    cgns, p2d = GRIDS / "n0012_113-33_hdf5.cgns", GRIDS / "n0012_113-33.p2dfmt"
    if not (cgns.is_file() and p2d.is_file()):
        pytest.skip("TMR grids (and the HDF5 conversion of the CGNS one) not on disk")
    (a,), _ = load_grid(cgns)
    (b,), _ = load_grid(p2d)
    assert a.sides == b.sides
    assert np.abs(np.asarray(a.x_node) - np.asarray(b.x_node)).max() < 1e-11
