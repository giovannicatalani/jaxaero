"""Multi-block grids round a high-lift section: a slat, a main element and a
flap, one behind the other, each trailing edge over the next one's nose.

What it takes (docs/multi-element.md): the elements' closed contours as
deployed, front to back, roles ``slat`` / ``main`` / ``flap`` in one of the
orders of ``FAMILIES``, coordinates in the case's length unit (a section of
order one); sharp or blunt trailing edges, smooth or stepped coves.
Configurations the template cannot mesh are refused with the cause, never
built folded.

Topology (the reasons, and what was measured, in docs/dev/notes.md):

- **Near-wall layer** round each element, one O-block marched from the wall
  (``generate.march``), ``i`` along the wall, ``j`` out, ``j = 0`` on the
  wall. Its thickness is ``layer_thickness``, capped by the distance to the
  other elements and by the wall's concave radius, slope-limited along the
  wall; the marching is smoothed into concave corners and the first cell
  re-marched to the wall spacing.
- **Fill blocks** between the layers' outer contours and a middle loop (a
  half-ellipse in front, a rectangle behind, ``MIDDLE_LOOP`` configuration
  chords out): ``front`` round the first element's nose; ``upper`` over it
  and its wake; per downstream element a ``slot`` over its upper surface and
  wake, under the upstream wake, and a ``cove`` from the upstream lip round
  to the gap; ``lower`` blocks under the elements, split at each cove lip.
  Each block is a Coons patch smoothed by Winslow's equations with spacing
  control (mesh/blocking.py); the cuts between blocks are fixed Hermite
  curves leaving each layer along its outward direction, and the wakes are
  offsets of the surface beneath them.
- **Outer ring**: the middle loop marched to ``far_field``.

Every vertex sits on a layer contour or the middle loop; every edge count is
a multiple of ``even_cells``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..geometry.elements import Element, chord_fraction, corners, cove_lip, nearest_on, surface, turning
from .blocking import (Block, arc_length, cell_areas, chain_nodes, chain_values, distribute, fill, hermite,
                       raw_blocks, smooth, solve_counts)
from .cgns import RawBlock
from .generate import geometric, march, march_to, vinokur

FAMILIES = {("main", "flap"): "main_flap", ("slat", "main"): "slat_main",
            ("slat", "main", "flap"): "slat_main_flap"}
"""Configurations the template takes, by the roles of their elements, front to back."""

SECTION_SIZE = (0.1, 10.0)
"""The section's extent along x, in the case's unit, must lie in this range
(coordinates in other units need ``coordinate_scale``)."""

LAYER_GAP_FRACTION = 0.3
"""A near-wall layer is at most this fraction of the distance to the nearest other element."""

LAYER_CONCAVE_FRACTION = 0.3
"""... and at most this fraction of the wall's concave radius, the turning measured over
two wall cells either side (the radius of a smooth wall; about a wall cell at a sharp corner)."""

LAYER_SLOPE = 0.2
"""Largest change of layer thickness per unit length along the wall."""

MARCH_SMOOTHING = 3.0
"""The marching's smoothing of its steps (``generate.march``'s default)."""

CONCAVE_SMOOTHING = 30.0
"""Its increase per right angle of concave turning near a node."""

FIRST_CELL_PASSES = 6
"""Marchings at most to bring every first cell within 1 % of the wall spacing."""

LAYER_RETRIES = 8
"""Times a folding layer is thinned round the fold and marched again."""

FRONT_FRACTION = 0.25
"""Chord fraction of the first element's points joined to the middle loop (upper and lower)."""

TE_CORNER_GROWTH = 4.0
"""At a blunt trailing edge's corners the wall cells are at most this many base cells."""

MOUTH_ARC = 0.5
"""A cove block's two long sides: the downstream nose (mouth foot to gap foot)
is this fraction of the upstream cove (lip to trailing edge)."""

MIDDLE_LOOP = 1.5
"""Middle loop: reach ahead, above, below and behind the configuration, in configuration chords."""

MIDDLE_LOOP_CELL = 8.0
"""The middle loop's spacing asked for, in wall cells (``1 / cells_per_chord``)."""

GROWTH = 1.15
"""Largest cell growth wanted along a cut, a connector or the outer ring."""

COVE_GROWTH = 1.3
"""Cell growth across a cove (from both its walls, so half the cells each way):
the gap shares the count, and the cove's recirculation needs no finer."""

OFFSET_SMOOTHING = 2.0
"""A wake offset's normals are smoothed over this many offset distances, so it
rounds a concave turn of its chain (a trailing edge's base) instead of creasing."""

WAKE_SPREAD = 0.1
"""A slot over a downstream element widens by this much per unit length downstream."""


@dataclass(frozen=True)
class Settings:
    """What the grid is asked to be (lengths in the case's unit)."""

    wall_spacing: float
    cells_per_chord: float
    layer_cells: int = 32
    layer_thickness: float = 0.02
    far_field: float = 100.0
    leading_edge_spacing: float = 2e-3
    trailing_edge_spacing: float = 2e-3
    gap_cells: int = 16
    even_cells: int = 4

    @property
    def nj(self) -> int:
        """Cells across a near-wall layer."""
        return _even(self.layer_cells, self.even_cells)


def family(roles: list[str]) -> str:
    """The family of a configuration with these roles, front to back.

    Raises:
        ValueError: for roles in an order the template does not take.
    """
    roles = tuple(roles)
    if roles not in FAMILIES:
        raise ValueError(f"no multi-element template for elements {list(roles)} (front to back); "
                         f"the templates take {[list(r) for r in FAMILIES]}")
    return FAMILIES[roles]


def _even(n: float, multiple: int) -> int:
    return int(-(-max(int(np.ceil(n)), 1) // multiple) * multiple)


def cells_for(length: float, ds0: float, ds1: float, growth: float = GROWTH) -> int:
    """Cells to go ``length`` from a first interval ``ds0`` to a last ``ds1``,
    growing by at most ``growth``: the geometric run from the smaller end,
    then uniform at the larger."""
    small, big = sorted((max(ds0, 1e-300), max(ds1, 1e-300)))
    n_grow = np.log(big / small) / np.log(growth)
    grown = small * (growth ** n_grow - 1.0) / (growth - 1.0)
    if grown >= length:
        return int(np.ceil(np.log(1.0 + length * (growth - 1.0) / small) / np.log(growth)))
    return int(np.ceil(n_grow + (length - grown) / big))


def _points(elements, skip: int | None = None) -> np.ndarray:
    """The contour points of the elements ``(n, 2)`` (but element ``skip``'s)."""
    return np.concatenate([np.stack([e.airfoil.x, e.airfoil.y], 1) for j, e in enumerate(elements) if j != skip])


# --- The wall loop of an element -------------------------------------------------------


@dataclass(frozen=True)
class _Wall:
    """An element's wall as a closed loop of arc length ``perimeter``: the
    contour (clockwise from the trailing edge's lower end, ``0 <= s <= L``)
    then the base back (blunt), and the landmarks along it."""

    element: Element
    landmarks: dict[str, float]          # name -> arc length, in loop order from s = 0

    @property
    def length(self) -> float:
        return self.element.airfoil.length

    @property
    def base(self) -> float:
        return 0.0 if self.element.airfoil.sharp else self.element.airfoil.trailing_edge_gap

    @property
    def perimeter(self) -> float:
        return self.length + self.base

    def points(self, s: np.ndarray) -> np.ndarray:
        a = self.element.airfoil
        s = np.asarray(s, dtype=float)
        p = surface(self.element)(np.minimum(s, self.length))
        if self.base:
            t = np.clip((s - self.length) / self.base, 0.0, 1.0)[..., None]
            end, start = np.array([a.x[-1], a.y[-1]]), np.array([a.x[0], a.y[0]])
            p = np.where((s > self.length)[..., None], end + t * (start - end), p)
        return p

    def at(self, name: str) -> np.ndarray:
        """A landmark's position."""
        return self.points(np.array([self.landmarks[name]]))[0]

    def pieces(self) -> list[tuple[str, str, float, float]]:
        """``(start, end, s0, s1)`` between consecutive landmarks; the last ends at the perimeter."""
        names = list(self.landmarks)
        s = [self.landmarks[n] for n in names] + [self.perimeter]
        return [(names[k], names[(k + 1) % len(names)], s[k], s[k + 1]) for k in range(len(names))]


def _surface_point(element: Element, fraction: float, upper: bool) -> float:
    """Arc length of the upper or lower surface point at ``fraction`` of the chord."""
    a = element.airfoil
    k = np.arange(a.leading_edge, a.x.size) if upper else np.arange(a.leading_edge + 1)[::-1]
    f = chord_fraction(element, np.stack([a.x[k], a.y[k]], axis=1))
    return float(np.interp(fraction, f, a.s[k]))


def _walls(elements: list[Element]) -> list[_Wall]:
    """Each element's landmarks: trailing edge (a blunt base's ends ``te_l``,
    ``te_u`` and middle ``tip``; a sharp one's ``tip``), leading edge ``le``,
    every corner (``k*``); the first element's front points ``a`` (upper)
    and ``b`` (lower; none when its lip is in its front half, as a slat's
    cusp is: the front block then reaches the lip); a cove lip ``lip`` on
    every element with a downstream one (the corner ``cove_lip`` finds, else
    the lower-surface point above the next nose) and on the last if it has
    one; a downstream element's gap foot ``g`` (nearest the upstream trailing
    edge) and mouth foot ``m`` (on its lower surface, the arc from ``m`` round
    its nose to ``g`` ``MOUTH_ARC`` of the upstream arc from lip to trailing
    edge).

    Raises:
        ValueError: if the landmarks are out of the order the template needs.
    """
    walls = []
    for k, e in enumerate(elements):
        a = e.airfoil
        s_le = float(a.s[a.leading_edge])
        if a.sharp:
            marks = {"tip": 0.0, "le": s_le}
        else:
            marks = {"te_l": 0.0, "le": s_le, "te_u": a.length, "tip": a.length + 0.5 * a.trailing_edge_gap}
        lip = cove_lip(e)
        if lip is not None:
            marks["lip"] = float(a.s[lip])
        elif k + 1 < len(elements):
            ahead = chord_fraction(e, elements[k + 1].leading_edge[None])[0]
            marks["lip"] = _surface_point(e, float(np.clip(ahead - 0.1, 0.5, 0.95)), upper=False)
        if k == 0:
            marks["a"] = _surface_point(e, FRONT_FRACTION, upper=True)
            if chord_fraction(e, _Wall(e, marks).at("lip")[None])[0] > 2.0 * FRONT_FRACTION:
                marks["b"] = _surface_point(e, FRONT_FRACTION, upper=False)
        else:
            marks["g"] = nearest_on(e, elements[k - 1].trailing_edge)[0]
            up = walls[-1]
            marks["m"] = marks["g"] - MOUTH_ARC * (up.landmarks["lip"] + 0.5 * up.base)
        taken = {round(v, 12) for v in marks.values()}
        for j, c in enumerate(corners(e)):         # every corner a node of the wall
            if round(float(a.s[c]), 12) not in taken:
                marks[f"k{j}"] = float(a.s[c])
        wall = _Wall(e, dict(sorted(marks.items(), key=lambda kv: kv[1])))
        order = [n for n in wall.landmarks if not n.startswith("k")]
        nose = ("g", "le") if "g" in marks and marks["g"] < marks["le"] else ("le", "g")   # g either side of le
        expected = [n for n in ("te_l", "lip", "m", "b", *nose, "a", "te_u") if n in marks]
        expected = ["tip"] + expected if a.sharp else expected + ["tip"]
        if order != expected:
            raise ValueError(f"the {e.role} element is outside the template's envelope: its landmarks lie "
                             f"in the order {order} round it, the template needs {expected}")
        walls.append(wall)
    return walls


# --- The block graph ------------------------------------------------------------------------


def _edge(k: int, a: str, b: str) -> str:
    return f"w{k}.{a}.{b}"


def _chain(wall_index: int, wall: _Wall, start: str, stop: str, forward: bool = True):
    """Layer edges of element ``wall_index`` from landmark ``start`` to ``stop``,
    walking clockwise (``forward``) or back."""
    names = list(wall.landmarks)
    out, k = [], names.index(start)
    step = 1 if forward else -1
    while names[k] != stop:
        nxt = (k + step) % len(names)
        a, b = (names[k], names[nxt]) if forward else (names[nxt], names[k])
        out.append((_edge(wall_index, a, b), not forward))
        k = nxt
    return out


def _gap_start(walls, k: int) -> str:
    """Where element ``k``'s gap cut leaves it: a blunt base's lower corner, else the trailing edge."""
    return "tip" if walls[k].element.airfoil.sharp else "te_l"


def _wake_start(walls, k: int) -> str:
    """Where element ``k``'s wake leaves it: a blunt base's upper corner (the
    upper surface runs straight on into the wake), the last element's base
    middle, or a sharp trailing edge."""
    return "tip" if walls[k].element.airfoil.sharp or k == len(walls) - 1 else "te_u"


def graph(walls: list[_Wall]) -> list[Block]:
    """The fill blocks: ``front`` round the first element's nose; ``upper``
    over the first element and its wake; per downstream element ``slot{k}``,
    over its upper surface and its wake, under the upstream wake, its left
    side the gap cut and the upstream base; ``cove{k}`` between the upstream
    lip, the gap cut, the downstream nose and the mouth; ``lower{k}`` under the
    elements, split at each lip by a cut down to the middle loop (``cut_b``
    from ``b``, or from the first lip if there is no ``b``; ``drop{k}`` from
    element ``k``'s lip), so three blocks meet at a lip, which on a slat is a
    cusp."""
    last = len(walls) - 1
    has_b = "b" in walls[0].landmarks
    blocks = [Block("front", tuple(_chain(0, walls[0], "b" if has_b else "lip", "a")), (("cut_a", False),),
                    (("ml_front", False),), (("cut_b", False),)),
              Block("upper", tuple(_chain(0, walls[0], "a", _wake_start(walls, 0)) + [("wake0", False)]),
                    (("ml_right0", False),), (("ml_top", False),), (("cut_a", False),))]
    for k in range(1, len(walls)):
        up, start = walls[k - 1], _gap_start(walls, k - 1)
        base = _chain(k - 1, up, start, _wake_start(walls, k - 1), forward=False)
        blocks.append(Block(f"slot{k}", tuple(_chain(k, walls[k], "g", _wake_start(walls, k)) + [(f"wake{k}", False)]),
                            ((f"ml_right{k}", False),), ((f"wake{k - 1}", False),),
                            tuple([(f"gap{k}", True)] + base)))
        blocks.append(Block(f"cove{k}", tuple(_chain(k - 1, up, "lip", start, forward=False)),
                            ((f"gap{k}", False),), tuple(_chain(k, walls[k], "m", "g")), ((f"mouth{k}", False),)))
    # Under the elements, a chain from each cut down to the next: from b (or
    # the first lip) and from every other lip.
    under, downs = [], ["cut_b"]
    for k, wall in enumerate(walls):
        head = [(f"mouth{k}", False)] if k else []
        start = "m" if k else "b"
        if "lip" in wall.landmarks and (k or has_b):
            under.append(head + _chain(k, wall, start, "lip", forward=False))
            downs.append(f"drop{k}")
            head, start = [], "lip"
        if k == last:
            under.append(head + _chain(k, wall, start, "tip", forward=False) + [(f"wake{last}", False)])
    downs.append("ml_right_lower")
    blocks += [Block(f"lower{k}", tuple(chain), ((right, False),), ((f"ml_bottom{k}", False),), ((left, False),))
               for k, (chain, left, right) in enumerate(zip(under, downs[:-1], downs[1:]))]
    return blocks


# --- Near-wall layers ------------------------------------------------------------------------


def _layer_thickness(walls, k: int, points: np.ndarray, s: np.ndarray,
                     settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Per wall node: ``layer_thickness`` capped at ``LAYER_GAP_FRACTION`` of
    the distance to the other elements and ``LAYER_CONCAVE_FRACTION`` of the
    concave radius, slope-limited along the loop; and how concave the wall is
    within ``layer_thickness`` of each node (its turning there over a right
    angle, at most 1)."""
    others = _points([w.element for w in walls], skip=k)
    d = np.min(np.hypot(points[:, None, 0] - others[None, :, 0], points[:, None, 1] - others[None, :, 1]), axis=1)
    t = np.minimum(settings.layer_thickness, LAYER_GAP_FRACTION * d)
    P = walls[k].perimeter
    ds = np.diff(np.concatenate([s, [s[0] + P]]))
    # Concave radius: the heading's change over +-w of arc (left turns are
    # concave on a clockwise loop), 2 w / turn.
    edge = np.roll(points, -1, axis=0) - points
    heading = np.unwrap(np.arctan2(edge[:, 1], edge[:, 0]))
    s_mid = s + 0.5 * ds
    lap = -2.0 * np.pi                       # a clockwise loop turns once, to the right
    ext_s = np.concatenate([s_mid - P, s_mid, s_mid + P])
    ext_h = np.concatenate([heading - lap, heading, heading + lap])
    w = 2.0 * np.maximum(ds, np.roll(ds, 1))
    turn = np.interp(s + w, ext_s, ext_h) - np.interp(s - w, ext_s, ext_h)
    concave = turn > 0.0
    t[concave] = np.minimum(t[concave], LAYER_CONCAVE_FRACTION * 2.0 * w[concave] / turn[concave])
    wide = settings.layer_thickness
    bend = np.interp(s + wide, ext_s, ext_h) - np.interp(s - wide, ext_s, ext_h)
    return _slope_limited(t, ds), np.clip(bend / (0.5 * np.pi), 0.0, 1.0)


def _slope_limited(t: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """``t`` lowered until it changes by at most ``LAYER_SLOPE`` per unit length
    round the periodic loop (``ds``: the interval after each node)."""
    t = t.copy()
    for _ in range(2):                                   # twice round: the loop is periodic
        for i in range(1, len(t) + 1):
            t[i % len(t)] = min(t[i % len(t)], t[i - 1] + LAYER_SLOPE * ds[i - 1])
        for i in range(len(t) - 2, -2, -1):
            t[i] = min(t[i], t[(i + 1) % len(t)] + LAYER_SLOPE * ds[i])
    return t


def _thick_enough(wall: _Wall, thick: float, where: np.ndarray, settings: Settings) -> None:
    """Raise ``ValueError`` if a layer of ``settings.nj`` cells from the wall spacing cannot be ``thick``."""
    if thick < 1.5 * settings.wall_spacing * settings.nj:
        raise ValueError(f"the {wall.element.role} element's near-wall layer would be {thick:.2e} thick at "
                         f"{np.round(where, 4)}: too thin for {settings.nj} cells from {settings.wall_spacing:g} "
                         "(a gap too narrow, or a corner too sharp)")


def _layer_estimate(walls, k: int, landmark: str, settings: Settings) -> tuple[float, float]:
    """A layer's thickness and last step at a landmark, before marching
    (``layer_thickness`` capped by the gap only)."""
    wall = walls[k]
    p = wall.at(landmark)
    others = _points([w.element for w in walls], skip=k)
    thick = min(settings.layer_thickness, LAYER_GAP_FRACTION * float(np.min(np.hypot(*(others - p).T))))
    _thick_enough(wall, thick, p, settings)
    return thick, float(np.diff(geometric(settings.nj, settings.wall_spacing, thick))[-1])


def _wall_nodes(walls, k: int, counts: dict, settings: Settings) -> tuple[np.ndarray, dict[str, int]]:
    """Wall nodes round the loop (the closing node not repeated) and each
    landmark's index. The spacing at a landmark: the leading-edge and
    trailing-edge spacings at ``le`` and at a sharp trailing edge or corner;
    at a blunt base's corners at most ``TE_CORNER_GROWTH`` base cells; where a
    cut leaves the layer along one block's side and the next block's chain
    (``g``, ``m``, ``lip``), the layer's last step there, so the cells either
    side of the cut match; elsewhere shared by the two pieces meeting there."""
    wall = walls[k]
    pieces = wall.pieces()
    n_base = counts.get(_edge(k, "te_u", "tip"), 0)
    base_cell = wall.base / (2 * n_base) if n_base else 0.0
    spacing = {}
    for n in wall.landmarks:
        if n == "le":
            spacing[n] = settings.leading_edge_spacing
        elif n in ("te_l", "te_u"):
            spacing[n] = min(settings.trailing_edge_spacing, TE_CORNER_GROWTH * base_cell)
        elif n in ("g", "m", "lip"):
            spacing[n] = _layer_estimate(walls, k, n, settings)[1]
        elif n.startswith("k") or n == "tip":
            spacing[n] = base_cell if n == "tip" and wall.base else settings.trailing_edge_spacing
    mean = {(a, b): (s1 - s0) / counts[_edge(k, a, b)] for a, b, s0, s1 in pieces}
    for n in wall.landmarks:
        if n not in spacing:
            spacing[n] = float(np.sqrt(np.prod([m for (a, b), m in mean.items() if n in (a, b)])))
    s_nodes, index = [], {}
    for a, b, s0, s1 in pieces:
        n = counts[_edge(k, a, b)]
        if "tip" in (a, b) and wall.base:                                  # a half base: uniform
            u = np.linspace(0.0, 1.0, n + 1)
        else:
            u = vinokur(n, spacing[a] / (s1 - s0), spacing[b] / (s1 - s0))
        index[a] = sum(len(p) for p in s_nodes)
        s_nodes.append(s0 + (s1 - s0) * u[:-1])
    s = np.concatenate(s_nodes)
    # Either side of a corner (or a sharp trailing edge), the nearer
    # neighbour's distance: marching steps a node normal to the chord through
    # its neighbours, along the corner's bisector only if they are equally far
    # (else a sharp corner's first cells twist; the slat cusp did, and a
    # sharp trailing edge, at any layer thickness).
    P = wall.perimeter
    for name, i in index.items():
        if name == "lip" or name.startswith("k") or (name == "tip" and not wall.base):
            prev, nxt = (i - 1) % len(s), (i + 1) % len(s)               # a sharp tip is node 0: wrap
            d = min((s[i] - s[prev]) % P, (s[nxt] - s[i]) % P)
            s[prev], s[nxt] = (s[i] - d) % P, (s[i] + d) % P
    return s, index


@dataclass(frozen=True)
class _Layer:
    nodes: np.ndarray                    # (n + 1, nj + 1, 2), the seam repeated
    index: dict[str, int]                # landmark -> node index (a landmark at s = 0: 0, or n as an end)

    def outer(self, name: str) -> np.ndarray:
        return self.nodes[self.index[name], -1]

    def outward(self, name: str) -> np.ndarray:
        d = self.nodes[self.index[name], -1] - self.nodes[self.index[name], -2]
        return d / np.hypot(*d)

    def step(self, name: str) -> float:
        return float(np.hypot(*(self.nodes[self.index[name], -1] - self.nodes[self.index[name], -2])))

    def next_steps(self) -> np.ndarray:
        """Per outer node, the layer's geometric progression one step further:
        the cell height a block built on the layer should start with."""
        last = np.hypot(*(self.nodes[:, -1] - self.nodes[:, -2]).T)
        before = np.hypot(*(self.nodes[:, -2] - self.nodes[:, -3]).T)
        return last * last / before

    def pieces(self, k: int, wall: _Wall):
        """``(edge, slice)`` of the outer contour between consecutive landmarks."""
        n = len(self.nodes) - 1
        for a, b, _, _ in wall.pieces():
            i0, i1 = self.index[a], self.index[b]
            yield _edge(k, a, b), slice(i0, (i1 if i1 > i0 else n) + 1)


def _smoothing(concave: np.ndarray, fan: np.ndarray, heights: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """The marching's smoothing per node and step ``(n, nj)``: from zero at the
    wall (the first steps orthogonal to it) to the default a wall cell out,
    ``1 + CONCAVE_SMOOTHING`` times it where the wall is concave (``concave``
    in [0, 1]: hyperbolic marching folds into a concave corner unless its
    steps are smoothed there; Chan and Steger, Appl. Math. Comput. 51 (1992)
    181-205); the default from the wall on where a convex corner fans
    (``fan``: the lines leaving it cross unless smoothed from the first step)."""
    ramp = np.clip(heights[:, :-1] / cell[:, None], 0.0, 1.0)
    wall = np.where(fan, MARCH_SMOOTHING, 0.0)[:, None]
    outer = MARCH_SMOOTHING * (1.0 + CONCAVE_SMOOTHING * concave[:, None])
    return wall + ramp * (outer - wall)


def _folded(nodes: np.ndarray) -> np.ndarray:
    """Wall-node indices of a periodic layer's columns holding a cell of the minority sign."""
    area = cell_areas(np.concatenate([nodes, nodes[:1]], axis=0))
    return np.flatnonzero(np.any(np.sign(area) != np.sign(np.median(area)), axis=1))


def _layer(walls, k: int, counts: dict, settings: Settings) -> _Layer:
    """Element ``k``'s layer, marched; where its cells fold (marching into a
    concave corner), the thickness is halved round the fold and the layer
    marched again; then the first cell re-marched to the wall spacing.

    Raises:
        ValueError: a layer too thin for its cells (a gap too narrow, or a
        fold that halving does not remove).
    """
    wall = walls[k]
    s, index = _wall_nodes(walls, k, counts, settings)
    line = wall.points(s)
    ds = np.diff(np.concatenate([s, [s[0] + wall.perimeter]]))
    t, concave = _layer_thickness(walls, k, line, s, settings)
    corner = np.flatnonzero(turning(line, closed=True) < -np.radians(30.0))   # convex: a sharp TE too
    fan = np.zeros(len(line), dtype=bool)
    fan[(corner[:, None] + np.arange(-2, 3)[None, :]).ravel() % len(line)] = True
    for _ in range(LAYER_RETRIES + 1):
        _thick_enough(wall, float(t.min()), line[np.argmin(t)], settings)
        heights = np.stack([geometric(settings.nj, settings.wall_spacing, ti) for ti in t])
        steps = np.diff(heights, axis=1)
        eps = _smoothing(concave, fan, heights, np.minimum(ds, np.roll(ds, 1)))
        nodes = march(line, steps, periodic=True, smoothing=eps)
        wrong = _folded(nodes)
        if not wrong.size:
            break
        near = np.unique(np.concatenate([(wrong + d) % len(t) for d in range(-2, 4)]))
        t[near] *= 0.5
        t = _slope_limited(t, ds)
    else:
        raise ValueError(f"the {wall.element.role} element's near-wall layer still folds after "
                         f"{LAYER_RETRIES} thinnings, at {np.round(line[wrong[0]], 4)}")
    # The marching's area smoothing moves the first cell off the wall spacing
    # where the wall spacing varies: corrected.
    for _ in range(FIRST_CELL_PASSES):
        first = np.hypot(*(nodes[:, 1] - nodes[:, 0]).T)
        if np.abs(first / settings.wall_spacing - 1.0).max() <= 0.01:
            break
        steps[:, 0] *= settings.wall_spacing / first
        corrected = march(line, steps, periodic=True, smoothing=eps)
        if _folded(corrected).size:
            break
        nodes = corrected
    if wall.base:                                  # the seam mid-base, between two equal cells
        shift = index["tip"]
        nodes = np.roll(nodes, -shift, axis=0)
        index = {name: (i - shift) % len(nodes) for name, i in index.items()}
    return _Layer(np.concatenate([nodes, nodes[:1]], axis=0), index)


# --- Counts ------------------------------------------------------------------------------------


def _requests(walls, settings: Settings, loop: _Loop) -> dict[str, int]:
    """Cells each edge asks for (before the counts are made consistent): wall
    pieces by ``cells_per_chord``; a blunt base, enough for the layer's outer
    contour round it (half a circle of the layer's thickness) to be spaced
    like the layer's last step; the middle loop's pieces by
    ``MIDDLE_LOOP_CELL``; each cut by growth at most ``GROWTH`` between the
    spacings at its ends (a gap at least ``gap_cells``, a mouth at
    ``COVE_GROWTH``); a wake over a slot as many as its slot's floor."""
    h_ml = MIDDLE_LOOP_CELL / settings.cells_per_chord
    est = {(k, n): _layer_estimate(walls, k, n, settings)
           for k, w in enumerate(walls) for n in ("tip", "lip", "g", "m", "a", "b") if n in w.landmarks}
    req = {}
    for k, wall in enumerate(walls):
        for a, b, s0, s1 in wall.pieces():
            if "tip" in (a, b) and wall.base:
                thick, step = est[(k, "tip")]
                req[_edge(k, a, b)] = 0.5 * np.pi * thick / step
            else:
                req[_edge(k, a, b)] = (s1 - s0) * settings.cells_per_chord
    for e, c in loop.curves().items():
        req[e] = float(arc_length(c)[-1]) / h_ml

    def cut(k0, n0, end, h_end, growth=GROWTH):
        return cells_for(float(np.hypot(*(end - walls[k0].at(n0)))), est[(k0, n0)][1], h_end, growth)

    last = len(walls) - 1
    for k in range(1, len(walls)):
        req[f"gap{k}"] = max(settings.gap_cells, cut(k - 1, "tip", walls[k].at("g"), est[(k, "g")][1]))
        req[f"mouth{k}"] = cut(k - 1, "lip", walls[k].at("m"), est[(k, "m")][1], COVE_GROWTH) // 2
        req[f"wake{k - 1}"] = 1
    for k, wall in enumerate(walls):
        if "lip" in wall.landmarks:
            req[f"drop{k}"] = cut(k, "lip", np.array([wall.at("lip")[0], loop.qb[1]]), h_ml)
    req["cut_a"] = cut(0, "a", loop.qa, h_ml)
    req["cut_b"] = cut(0, "b" if "b" in walls[0].landmarks else "lip", loop.qb, h_ml)
    req[f"wake{last}"] = cut(last, "tip", np.array([loop.rt[0], walls[last].at("tip")[1]]), h_ml)
    return req


# --- The middle loop ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Loop:
    """The middle loop: a half-ellipse ahead of ``qb``, ``qa``, a rectangle
    behind to ``rt``, ``rb``; ``feet`` on the bottom where the drops arrive,
    ``wakes`` on the right where the wakes arrive (top to bottom)."""

    qa: np.ndarray
    qb: np.ndarray
    rt: np.ndarray
    rb: np.ndarray
    centre: np.ndarray
    reach: float
    feet: tuple[np.ndarray, ...]
    wakes: tuple[np.ndarray, ...]

    def curves(self) -> dict[str, np.ndarray]:
        t = np.radians(np.linspace(270.0, 90.0, 400))
        half = 0.5 * (self.qa[1] - self.qb[1])
        front = np.stack([self.centre[0] + self.reach * np.cos(t), self.centre[1] + half * np.sin(t)], 1)
        front[0], front[-1] = self.qb, self.qa
        bottom = [self.qb, *self.feet, self.rb]
        right = [self.rt, *self.wakes]
        return {"ml_front": front, "ml_top": np.array([self.qa, self.rt]),
                **{f"ml_bottom{k}": np.array([p, q]) for k, (p, q) in enumerate(zip(bottom[:-1], bottom[1:]))},
                **{f"ml_right{k}": np.array([q, p]) for k, (p, q) in enumerate(zip(right[:-1], right[1:]))},
                "ml_right_lower": np.array([self.wakes[-1], self.rb])}


def _middle_loop(walls, front_x: float, drop_x: list[float]) -> _Loop:
    """The loop round the walls, its wake points provisional (evenly down the right side's middle).

    Raises:
        ValueError: if the lips are not in order behind the front.
    """
    pts = _points([w.element for w in walls])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = MIDDLE_LOOP * (hi[0] - lo[0])
    yc = 0.5 * (lo[1] + hi[1])
    x_back = hi[0] + size
    if not front_x < min(drop_x + [x_back]) or np.any(np.diff(drop_x) <= 0.0):
        raise ValueError(f"the lips are not in order behind the front of the section (x {np.round(drop_x, 4)}, "
                         f"front {front_x:.4f}): outside the template's envelope")
    wakes = [np.array([x_back, y]) for y in np.linspace(yc + 0.25 * size, yc - 0.25 * size, len(walls))]
    return _Loop(np.array([front_x, yc + size]), np.array([front_x, yc - size]), np.array([x_back, yc + size]),
                 np.array([x_back, yc - size]), np.array([front_x, yc]), front_x - (lo[0] - size),
                 tuple(np.array([x, yc - size]) for x in drop_x), tuple(wakes))


def _distribute_loop(curves: dict[str, np.ndarray], counts: dict[str, int]) -> dict[str, np.ndarray]:
    """Middle-loop edges by Vinokur, the spacing at each corner shared by the
    edges meeting there (the geometric mean of their means)."""
    ends = {e: (tuple(np.round(c[0], 12)), tuple(np.round(c[-1], 12))) for e, c in curves.items()}
    corner = {}
    for e, c in curves.items():
        for v in ends[e]:
            corner.setdefault(v, []).append(float(arc_length(c)[-1]) / counts[e])
    ds = {v: float(np.exp(np.mean(np.log(m)))) for v, m in corner.items()}
    return {e: distribute(c, counts[e], ds[ends[e][0]], ds[ends[e][1]]) for e, c in curves.items()}


# --- Cuts and wakes ----------------------------------------------------------------------------


def _offset(chain: np.ndarray, start: np.ndarray, x_back: float) -> np.ndarray:
    """A wake over a downstream element: its slot's bottom ``chain`` (the
    element's upper contour and wake, nodes ``(n, 2)``) moved node for node
    to the left by ``d = d0 + WAKE_SPREAD s`` (``d0`` the distance from its
    first node to ``start``, ``s`` the arc length), along normals smoothed
    over ``OFFSET_SMOOTHING d``, the result smoothed over ``d``; the first
    ``4 d0`` blended onto ``start``; the last node on ``x = x_back``.

    Raises:
        ValueError: if the offset doubles back on itself.
    """
    s = arc_length(chain)
    tangent = np.gradient(chain, s, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    d0 = float(np.hypot(*(start - chain[0])))
    d = d0 + WAKE_SPREAD * s
    weight = np.exp(-0.5 * ((s[:, None] - s[None, :]) / (OFFSET_SMOOTHING * d[:, None])) ** 2)
    normal = weight @ normal
    normal /= np.hypot(*normal.T)[:, None]
    out = chain + d[:, None] * normal
    # A corner of the chain (a trailing edge's base, then its wake) shifts
    # with the normals unrounded: the offset itself is smoothed over d too.
    # (Averaging moves unevenly spaced nodes along the curve too: the smoothed
    # curve keeps the shape, the nodes go back to the offset's arc fractions.)
    weight = np.exp(-0.5 * ((s[:, None] - s[None, :]) / d[:, None]) ** 2)
    shape = weight @ out / weight.sum(axis=1)[:, None]
    shape[0], shape[-1] = out[0], out[-1]
    u_out, u_shape = (arc_length(c) / arc_length(c)[-1] for c in (out, shape))
    out = np.stack([np.interp(u_out, u_shape, shape[:, 0]), np.interp(u_out, u_shape, shape[:, 1])], axis=1)
    blend = np.clip(1.0 - s / (4.0 * d0), 0.0, 1.0)[:, None] ** 2
    out += blend * (start - out[0])
    out[-1, 0] = x_back
    along = np.stack([normal[:, 1], -normal[:, 0]], axis=1)            # the smoothed tangent
    if np.any(np.sum(np.diff(out, axis=0) * 0.5 * (along[1:] + along[:-1]), axis=1) <= 0.0):
        raise ValueError("a wake offset over a downstream element doubles back: outside the template's envelope")
    return out


def _cuts(walls, layers, blocks, counts, loop: _Loop, h_ml: float, size: float) -> dict[str, np.ndarray]:
    """The edges between fill blocks: gaps, mouths, drops, the front cuts and
    the wakes, each a Hermite curve leaving a layer along its outward
    direction, distributed to the spacing at its ends; the wakes over a
    downstream element offsets of its slot's floor (``_offset``), built back
    to front."""
    nodes: dict[str, np.ndarray] = {}

    def cut(k0, v0, n_cells, k1=None, v1=None, end=None, end_dir=None, end_step=None, lean=None):
        """From layer ``k0``'s vertex ``v0`` along its outward direction (turned
        half way to ``lean``) to layer ``k1``'s vertex ``v1``, or to ``end``
        arriving along ``end_dir``."""
        out0 = layers[k0].outward(v0)
        if lean is not None:
            out0 = out0 + lean / np.hypot(*lean)
        if end is None:
            end, end_dir, end_step = layers[k1].outer(v1), -layers[k1].outward(v1), layers[k1].step(v1)
        return distribute(hermite(layers[k0].outer(v0), out0, end, end_dir), n_cells, layers[k0].step(v0), end_step)

    down, last = np.array([0.0, -1.0]), len(walls) - 1
    front = "b" if "b" in walls[0].landmarks else "lip"
    edges = set().union(*(b.edges for b in blocks))
    for k in range(1, len(walls)):
        nodes[f"gap{k}"] = cut(k - 1, _gap_start(walls, k - 1), counts[f"gap{k}"], k, "g")
        # the mouth and the drop leave a lip together, leaning apart
        lean = layers[k].outer("m") - layers[k - 1].outer("lip")
        nodes[f"mouth{k}"] = cut(k - 1, "lip", counts[f"mouth{k}"], k, "m", lean=lean)
    drops = [k for k in range(len(walls)) if f"drop{k}" in edges]
    for foot, k in zip(loop.feet, drops):
        nodes[f"drop{k}"] = cut(k, "lip", counts[f"drop{k}"], end=foot, end_dir=down, end_step=h_ml, lean=down)
    nodes["cut_a"] = cut(0, "a", counts["cut_a"], end=loop.qa, end_dir=-down, end_step=h_ml)
    nodes["cut_b"] = cut(0, front, counts["cut_b"], end=loop.qb, end_dir=down, end_step=h_ml,
                         lean=None if front == "b" else down)
    tip = layers[last].outer("tip")
    y_end = float(np.clip(tip[1], loop.centre[1] - 0.5 * size, loop.centre[1] + 0.5 * size))
    nodes[f"wake{last}"] = cut(last, "tip", counts[f"wake{last}"], end=np.array([loop.rt[0], y_end]),
                               end_dir=np.array([1.0, 0.0]), end_step=h_ml)
    return nodes


def _wakes(walls, layers, blocks, nodes: dict[str, np.ndarray], loop: _Loop) -> None:
    """The wakes over the downstream elements, back to front, into ``nodes``."""
    slots = {b.name: b for b in blocks}
    for k in range(len(walls) - 2, -1, -1):
        floor = chain_nodes(slots[f"slot{k + 1}"].bottom, nodes)
        nodes[f"wake{k}"] = _offset(floor, layers[k].outer(_wake_start(walls, k)), loop.rt[0])


def _off_wall(blocks, nodes, layers, walls, h_ml) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per fill block, the cell heights wanted off its bottom and top sides:
    a layer's next step on a layer contour, the middle loop's spacing on the
    loop, and along a cut the geometric interpolation of its two ends' (the
    same for the blocks either side of it)."""
    off, ends = {}, {}
    for k, (wall, layer) in enumerate(zip(walls, layers)):
        steps = layer.next_steps()
        for e, piece in layer.pieces(k, wall):
            off[e] = steps[piece]
            for i in (piece.start, piece.stop - 1):
                ends[tuple(np.round(layer.nodes[i, -1], 12))] = steps[i]
    for e, c in nodes.items():
        if e.startswith("ml_"):
            off[e] = np.full(len(c), h_ml)
        elif e not in off:                                   # a cut
            h0, h1 = (ends.get(tuple(np.round(c[k], 12)), h_ml) for k in (0, -1))
            u = arc_length(c)
            off[e] = h0 * (h1 / h0) ** (u / u[-1])
    return {b.name: (chain_values(b.bottom, off), chain_values(b.top, off)) for b in blocks}


def _ring(nodes, loop: _Loop, walls, settings: Settings, h_ml: float, size: float) -> np.ndarray:
    """The outer ring: the middle loop, clockwise from the top front corner,
    marched to ``far_field``; ``(n + 1, nj + 1, 2)``, the seam repeated."""
    right = [nodes[f"ml_right{k}"][::-1][:-1] for k in range(len(walls))]
    bottom = [nodes[f"ml_bottom{k}"][::-1][:-1] for k in reversed(range(len(loop.feet) + 1))]
    line = np.concatenate([nodes["ml_top"][:-1], *right, nodes["ml_right_lower"][:-1], *bottom,
                           nodes["ml_front"][:-1]])
    reach = settings.far_field - size
    cells = _even(cells_for(reach, h_ml, reach, 1.2), settings.even_cells)
    ring = march_to(line, _points([w.element for w in walls]), cells, h_ml, settings.far_field, periodic=True)
    return np.concatenate([ring, ring[:1]], axis=0)


# --- The grid ------------------------------------------------------------------------------------


def multi_element(elements: list[Element], settings: Settings) -> tuple[list[RawBlock], dict]:
    """The grid's blocks (layers, fill blocks, outer ring) and a record of
    what was built (family, blocks, cells).

    Raises:
        ValueError: a configuration the template does not take (with the
        cause), or a grid that folds.
    """
    name = family([e.role for e in elements])
    extent = float(np.ptp(_points(elements)[:, 0]))
    if not SECTION_SIZE[0] <= extent <= SECTION_SIZE[1]:
        raise ValueError(f"the section spans {extent:.4g} along x: coordinates in another unit than the case's "
                         f"(the Reynolds number's length)? Set domain.coordinate_scale")
    walls = _walls(elements)
    blocks = graph(walls)
    size = MIDDLE_LOOP * extent
    edges = set().union(*(b.edges for b in blocks))
    front = "b" if "b" in walls[0].landmarks else "lip"
    drops = [k for k in range(len(walls)) if f"drop{k}" in edges]

    def loop_through(point):
        """The middle loop placed by the vertices (``point(k, name)``)."""
        return _middle_loop(walls, 0.5 * (point(0, "a")[0] + point(0, front)[0]),
                            [float(point(k, "lip")[0]) for k in drops])

    # Counts first, from the walls (the loop placed by the wall landmarks).
    requests = _requests(walls, settings, loop_through(lambda k, v: walls[k].at(v)))
    counts = solve_counts(blocks, {e: n for e, n in requests.items() if e in edges}, settings.even_cells)

    layers = [_layer(walls, k, counts, settings) for k in range(len(walls))]
    nodes = {e: layer.nodes[piece, -1] for k, (wall, layer) in enumerate(zip(walls, layers))
             for e, piece in layer.pieces(k, wall)}
    loop = loop_through(lambda k, v: layers[k].outer(v))
    h_ml = float(np.mean([arc_length(c)[-1] / counts[e] for e, c in loop.curves().items()
                          if not e.startswith("ml_right")]))
    nodes.update(_cuts(walls, layers, blocks, counts, loop, h_ml, size))
    _wakes(walls, layers, blocks, nodes, loop)
    loop = replace(loop, wakes=tuple(nodes[f"wake{k}"][-1] for k in range(len(walls))))
    if not loop.rt[1] > loop.wakes[0][1]:
        raise ValueError("the wakes reach above the middle loop: outside the template's envelope")
    nodes.update(_distribute_loop(loop.curves(), counts))

    grids = smooth(fill(blocks, nodes), _off_wall(blocks, nodes, layers, walls, h_ml))
    ring = _ring(nodes, loop, walls, settings, h_ml, size)

    raw = [RawBlock(layer.nodes[..., 0].copy(), layer.nodes[..., 1].copy(), name=f"layer{k}")
           for k, layer in enumerate(layers)]
    raw += raw_blocks(grids)
    raw.append(RawBlock(ring[..., 0].copy(), ring[..., 1].copy(), name="ring"))
    cells = int(sum((b.x.shape[0] - 1) * (b.x.shape[1] - 1) for b in raw))
    return raw, {"family": name, "blocks": len(raw), "cells": cells}
