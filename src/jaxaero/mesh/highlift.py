"""Multi-block grids round a high-lift section: elements one behind the other
(slat, main element, flap), each trailing edge over the next one's nose.

Topology (the reasons, and what was measured, in docs/dev/notes.md):

- **Near-wall layer** round each element, one O-block marched from the wall
  (``generate.march``), ``i`` along the wall, ``j`` out, ``j = 0`` on the
  wall. Its thickness is ``layer_thickness``, capped at ``LAYER_GAP_FRACTION``
  of the distance to the nearest other element (so the layers leave room in
  the gaps), with a bounded slope along the wall.
- **Fill blocks** between the layers' outer contours and a middle loop (a
  half-ellipse in front, a rectangle behind, ``MIDDLE_LOOP`` configuration
  chords out): a front block wrapping the first element's nose; one block
  over the whole chain (upper surfaces, the gap cuts from each trailing edge
  to the next element, the wake cut from the last trailing edge); one block
  under it (lower surfaces, the cove mouths, the wake cut); and per pair of
  elements a cove block (the upstream element's lip to its trailing edge,
  the gap cut, the downstream element's nose, the mouth back to the lip).
  Transfinite interpolation, then elliptic smoothing of each (mesh/blocking.py);
  the cuts between them are Hermite curves leaving each vertex along its
  layer's outward direction, distributed to match the spacing at their ends.
- **Outer ring**: the middle loop marched to ``far_field``.

The three families are compositions of the same pieces: a front for the
first element, a gap module per pair, a wake for the last. Every vertex sits
on a layer contour or the middle loop; every edge count is a multiple of
``even_cells``. A configuration whose landmarks are not in the order the
template needs is refused, naming what is out of place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.elements import Element, chord_fraction, corners, cove_lip, nearest_on, surface
from .blocking import (Block, arc_length, cell_areas, chain_nodes, chain_values, distribute, fill, hermite, raw_blocks,
                       smooth, solve_counts)
from .cgns import RawBlock
from .generate import geometric, march, march_to, vinokur

FAMILIES = {("main", "flap"): "main_flap", ("slat", "main"): "slat_main",
            ("slat", "main", "flap"): "slat_main_flap"}
"""Configurations the template takes, by the roles of their elements, front to back."""

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

    def pieces(self) -> list[tuple[str, str, float, float]]:
        """``(start, end, s0, s1)`` between consecutive landmarks; the last ends at the perimeter."""
        names = list(self.landmarks)
        s = [self.landmarks[n] for n in names] + [self.perimeter]
        return [(names[k], names[(k + 1) % len(names)], s[k], s[k + 1]) for k in range(len(names))]


def _lower_point(element: Element, fraction: float) -> float:
    """Arc length of the lower-surface point at ``fraction`` of the chord."""
    a = element.airfoil
    k = np.arange(a.leading_edge + 1)
    f = chord_fraction(element, np.stack([a.x[k], a.y[k]], axis=1))
    return float(np.interp(fraction, f[::-1], a.s[k][::-1]))


def _upper_point(element: Element, fraction: float) -> float:
    a = element.airfoil
    k = np.arange(a.leading_edge, a.x.size)
    f = chord_fraction(element, np.stack([a.x[k], a.y[k]], axis=1))
    return float(np.interp(fraction, f, a.s[k]))


def _walls(elements: list[Element]) -> list[_Wall]:
    """Each element's landmarks: trailing edge (its base's ends and middle,
    ``tip``), leading edge; the first element's front points ``a`` (upper)
    and ``b`` (lower; none when its lip is in its front half, as a slat's
    cusp is: the front block then reaches the lip); an upstream element's cove
    lip (``lip``; the last element's only if it has one); a downstream
    element's gap foot (``g``, nearest the upstream
    trailing edge) and mouth foot (``m``, on its lower surface, the arc from
    ``m`` round its nose to ``g`` ``MOUTH_ARC`` of the upstream arc from lip to
    trailing edge).

    Raises:
        ValueError: if the landmarks are out of the order the template needs.
    """
    walls = []
    for k, e in enumerate(elements):
        a = e.airfoil
        s_le = float(a.s[a.leading_edge])
        marks = {"te_l": 0.0, "le": s_le, "te_u": a.length} if not a.sharp else {"tip": 0.0, "le": s_le}
        if not a.sharp:
            marks["tip"] = a.length + 0.5 * a.trailing_edge_gap
        lip = cove_lip(e)
        if lip is not None:
            marks["lip"] = float(a.s[lip])
        elif k + 1 < len(elements):     # no corner: the lower-surface point above the next element's nose
            ahead = chord_fraction(e, elements[k + 1].leading_edge[None])[0]
            marks["lip"] = _lower_point(e, float(np.clip(ahead - 0.1, 0.5, 0.95)))
        if k == 0:
            marks["a"] = _upper_point(e, FRONT_FRACTION)
            f_lip = chord_fraction(e, _Wall(e, {}).points(np.array([marks["lip"]])))[0]
            if f_lip > 2.0 * FRONT_FRACTION:
                marks["b"] = _lower_point(e, FRONT_FRACTION)
        else:
            marks["g"] = nearest_on(e, elements[k - 1].trailing_edge)[0]
            up = walls[-1]
            cove = up.landmarks["lip"] + 0.5 * up.base               # lip back to the tip
            marks["m"] = marks["g"] - MOUTH_ARC * cove
        taken = {round(v, 12) for v in marks.values()}
        for j, c in enumerate(corners(e)):         # every corner a node of the wall
            if round(float(a.s[c]), 12) not in taken:
                marks[f"k{j}"] = float(a.s[c])
        wall = _Wall(e, dict(sorted(marks.items(), key=lambda kv: kv[1])))
        order = [n for n in wall.landmarks if not n.startswith("k")]
        nose = ("g", "le") if "g" in marks and marks["g"] < marks["le"] else ("le", "g")   # g either side of le
        expected = [n for n in ("te_l", "tip", "lip", "m", "b", *nose, "a", "te_u") if n in marks]
        if a.sharp:
            expected = ["tip"] + [n for n in expected if n != "tip"]
        else:
            expected = [n for n in expected if n != "tip"] + ["tip"]
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


# --- Geometry -----------------------------------------------------------------------------


def _layer_thickness(walls, k: int, points: np.ndarray, s: np.ndarray, settings: Settings) -> np.ndarray:
    """Per wall node: ``layer_thickness`` capped at ``LAYER_GAP_FRACTION`` of
    the distance to the other elements and ``LAYER_CONCAVE_FRACTION`` of the
    concave radius, slope-limited along the loop; and how concave the wall is
    within ``layer_thickness`` of each node (its turning there over a right
    angle, at most 1)."""
    others = np.concatenate([np.stack([w.element.airfoil.x, w.element.airfoil.y], 1)
                             for j, w in enumerate(walls) if j != k] or [np.full((1, 2), np.inf)])
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


def _landmark_spacing(wall: _Wall, name: str, settings: Settings, base_cells: int) -> float | None:
    """Wall spacing at a landmark; ``None``: shared by the pieces meeting there."""
    base_cell = wall.base / (2 * base_cells) if base_cells else 0.0
    if name == "le":
        return settings.leading_edge_spacing
    if name in ("te_l", "te_u"):              # at most TE_CORNER_GROWTH times the base's cells
        return min(settings.trailing_edge_spacing, TE_CORNER_GROWTH * base_cell)
    if name in ("lip", "g") or name.startswith("k") or (name == "tip" and wall.element.airfoil.sharp):
        return settings.trailing_edge_spacing
    if name == "tip":
        return base_cell
    return None


def _layer_estimate(walls, k: int, landmark: str, settings: Settings) -> tuple[float, float]:
    """A layer's thickness and last step at a landmark, before marching
    (``layer_thickness`` capped by the gap only)."""
    wall = walls[k]
    p = wall.points(np.array([wall.landmarks[landmark]]))[0]
    others = np.concatenate([np.stack([w.element.airfoil.x, w.element.airfoil.y], 1)
                             for j, w in enumerate(walls) if j != k])
    thick = min(settings.layer_thickness, LAYER_GAP_FRACTION * float(np.min(np.hypot(*(others - p).T))))
    nj = _even(settings.layer_cells, settings.even_cells)
    _thick_enough(wall, thick, p, nj, settings)
    return thick, float(np.diff(geometric(nj, settings.wall_spacing, thick))[-1])


def _thick_enough(wall: _Wall, thick: float, where: np.ndarray, nj: int, settings: Settings) -> None:
    """Raise ``ValueError`` if a layer of ``nj`` cells from the wall spacing cannot be ``thick``."""
    if thick < 1.5 * settings.wall_spacing * nj:
        raise ValueError(f"the {wall.element.role} element's near-wall layer would be {thick:.2e} thick at "
                         f"{np.round(where, 4)}: too thin for {nj} cells from {settings.wall_spacing:g} "
                         "(a gap too narrow, or a corner too sharp)")


def _wall_nodes(walls, k: int, counts: dict, settings: Settings) -> tuple[np.ndarray, dict[str, int]]:
    """Wall nodes round the loop (the closing node not repeated) and each
    landmark's index. Where a cut leaves the layer along a block's side and
    runs along the next block's chain (``g``, ``m``, ``lip``), the wall spacing
    is the layer's last step there (estimated from the gap cap), so the cells
    either side of the cut match."""
    wall = walls[k]
    pieces = wall.pieces()
    n_base = counts.get(_edge(k, "te_u", "tip"), 0)
    spacing = {n: _landmark_spacing(wall, n, settings, n_base) for n in wall.landmarks}
    for n in ("g", "m", "lip"):
        if n in wall.landmarks:
            spacing[n] = _layer_estimate(walls, k, n, settings)[1]
    mean = {(a, b): (s1 - s0) / counts[_edge(k, a, b)] for a, b, s0, s1 in pieces}
    for n in spacing:                        # unset: shared by the two pieces meeting there
        if spacing[n] is None:
            near = [m for (a, b), m in mean.items() if n in (a, b)]
            spacing[n] = float(np.sqrt(np.prod(near)))
    s_nodes, index = [], {}
    for a, b, s0, s1 in pieces:
        n = counts[_edge(k, a, b)]
        if "tip" in (a, b) and not wall.element.airfoil.sharp:            # a half base: uniform
            u = np.linspace(0.0, 1.0, n + 1)
        else:
            u = vinokur(n, spacing[a] / (s1 - s0), spacing[b] / (s1 - s0))
        index[a] = sum(len(p) for p in s_nodes)
        s_nodes.append(s0 + (s1 - s0) * u[:-1])
    s = np.concatenate(s_nodes)
    # Either side of a corner, the nearer neighbour's distance: marching steps a
    # node normal to the chord through its neighbours, along the corner's
    # bisector only if they are equally far (else a sharp corner's first
    # cells twist; the slat cusp did, at any layer thickness).
    for name, i in index.items():
        if name == "lip" or name.startswith("k"):
            d = min(s[i] - s[i - 1], s[i + 1] - s[i])
            s[i - 1], s[i + 1] = s[i] - d, s[i] + d
    return s, index


@dataclass(frozen=True)
class _Layer:
    nodes: np.ndarray                    # (n + 1, nj + 1, 2), the seam repeated
    index: dict[str, int]                # landmark -> node index (a landmark at s = 0: 0, or n as an end)

    def outer(self, i: int) -> np.ndarray:
        return self.nodes[i, -1]

    def outward(self, i: int) -> np.ndarray:
        d = self.nodes[i, -1] - self.nodes[i, -2]
        return d / np.hypot(*d)

    def step(self, i: int) -> float:
        return float(np.hypot(*(self.nodes[i, -1] - self.nodes[i, -2])))

    def next_steps(self) -> np.ndarray:
        """Per outer node, the layer's geometric progression one step further:
        the cell height a block built on the layer should start with."""
        last = np.hypot(*(self.nodes[:, -1] - self.nodes[:, -2]).T)
        before = np.hypot(*(self.nodes[:, -2] - self.nodes[:, -3]).T)
        return last * last / before


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


def _layers(walls, counts, settings: Settings) -> list[_Layer]:
    """Each element's layer, marched; where its cells fold (marching into a
    concave corner), the thickness is halved round the fold and the layer
    marched again.

    Raises:
        ValueError: a layer too thin for its cells (a gap too narrow, or a
        fold that halving does not remove).
    """
    nj = _even(settings.layer_cells, settings.even_cells)
    out = []
    for k, wall in enumerate(walls):
        s, index = _wall_nodes(walls, k, counts, settings)
        line = wall.points(s)
        ds = np.diff(np.concatenate([s, [s[0] + wall.perimeter]]))
        t, concave = _layer_thickness(walls, k, line, s, settings)
        edge = np.roll(line, -1, axis=0) - line
        heading = np.unwrap(np.arctan2(edge[:, 1], edge[:, 0]))
        turn = heading - np.roll(heading, 1)
        turn[0] += 2.0 * np.pi                             # the seam: the loop's lap
        corner = np.flatnonzero(turn < -np.radians(30.0))  # convex corners, to the right
        fan = np.zeros(len(line), dtype=bool)
        fan[(corner[:, None] + np.arange(-2, 3)[None, :]).ravel() % len(line)] = True
        for _ in range(LAYER_RETRIES + 1):
            _thick_enough(wall, float(t.min()), line[np.argmin(t)], nj, settings)
            heights = np.stack([geometric(nj, settings.wall_spacing, ti) for ti in t])
            steps = np.diff(heights, axis=1)
            eps = _smoothing(concave, fan, heights, np.minimum(ds, np.roll(ds, 1)))
            nodes = march(line, steps, periodic=True, smoothing=eps)
            wrong = _folded(nodes)
            if not wrong.size:
                # The marching's area smoothing moves the first cell off the
                # wall spacing where the wall spacing varies: corrected.
                for _ in range(FIRST_CELL_PASSES):
                    first = np.hypot(*(nodes[:, 1] - nodes[:, 0]).T)
                    if np.abs(first / settings.wall_spacing - 1.0).max() <= 0.01:
                        break
                    steps[:, 0] *= settings.wall_spacing / first
                    corrected = march(line, steps, periodic=True, smoothing=eps)
                    if _folded(corrected).size:
                        break
                    nodes = corrected
                break
            near = np.unique(np.concatenate([(wrong + d) % len(t) for d in range(-2, 4)]))
            t[near] *= 0.5
            t = _slope_limited(t, ds)
        else:
            raise ValueError(f"the {wall.element.role} element's near-wall layer still folds after "
                             f"{LAYER_RETRIES} thinnings, at {np.round(line[wrong[0]], 4)}")
        if not wall.element.airfoil.sharp:      # the seam mid-base, between two equal cells
            shift = index["tip"]
            nodes = np.roll(nodes, -shift, axis=0)
            index = {name: (i - shift) % len(nodes) for name, i in index.items()}
        out.append(_Layer(np.concatenate([nodes, nodes[:1]], axis=0), index))
    return out


def _folded(nodes: np.ndarray) -> np.ndarray:
    """Wall-node indices of a periodic layer's columns holding a cell of the minority sign."""
    area = cell_areas(np.concatenate([nodes, nodes[:1]], axis=0))
    return np.flatnonzero(np.any(np.sign(area) != np.sign(np.median(area)), axis=1))


def _requests(walls, settings: Settings, loop: "_Loop") -> dict[str, int]:
    """Cells each edge asks for (before the counts are made consistent): wall
    pieces by ``cells_per_chord``; a blunt base, enough for the layer's outer
    contour round it (half a circle of the layer's thickness) to be spaced
    like the layer's last step; the middle loop's pieces by
    ``MIDDLE_LOOP_CELL``; each cut by growth at most ``GROWTH`` between the
    spacings at its ends (a gap at least ``gap_cells``)."""
    h_ml = MIDDLE_LOOP_CELL / settings.cells_per_chord
    est = {(k, n): _layer_estimate(walls, k, n, settings)
           for k, w in enumerate(walls) for n in ("tip", "lip", "g", "m", "a", "b") if n in w.landmarks}
    at = lambda k, n: walls[k].points(np.array([walls[k].landmarks[n]]))[0]  # noqa: E731
    req = {}
    for k, wall in enumerate(walls):
        for a, b, s0, s1 in wall.pieces():
            if "tip" in (a, b) and not wall.element.airfoil.sharp:
                thick, step = est[(k, "tip")]
                req[_edge(k, a, b)] = 0.5 * np.pi * thick / step
            else:
                req[_edge(k, a, b)] = (s1 - s0) * settings.cells_per_chord
    for e, c in loop.curves().items():
        req[e] = np.sum(np.hypot(*np.diff(c, axis=0).T)) / h_ml

    def cut(k0, n0, end, h_end):
        return cells_for(float(np.hypot(*(end - at(k0, n0)))), est[(k0, n0)][1], h_end)

    front = "b" if "b" in walls[0].landmarks else "lip"
    last = len(walls) - 1
    for k in range(1, len(walls)):
        req[f"gap{k}"] = max(settings.gap_cells, cells_for(float(np.hypot(*(at(k, "g") - at(k - 1, "tip")))),
                                                           est[(k - 1, "tip")][1], est[(k, "g")][1]))
        req[f"mouth{k}"] = cells_for(float(np.hypot(*(at(k, "m") - at(k - 1, "lip")))), est[(k - 1, "lip")][1],
                                     est[(k, "m")][1], COVE_GROWTH) // 2


        req[f"wake{k - 1}"] = 1                       # an offset of the next slot's chain: its count
    for k, wall in enumerate(walls):
        if "lip" in wall.landmarks:
            req[f"drop{k}"] = cut(k, "lip", np.array([at(k, "lip")[0], loop.qb[1]]), h_ml)
    req["cut_a"] = cut(0, "a", loop.qa, h_ml)
    req["cut_b"] = cut(0, front, loop.qb, h_ml)
    req[f"wake{last}"] = cut(last, "tip", np.array([loop.rt[0], at(last, "tip")[1]]), h_ml)
    return req


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

    def with_wakes(self, wakes) -> _Loop:
        return _Loop(self.qa, self.qb, self.rt, self.rb, self.centre, self.reach, self.feet, tuple(wakes))


def _middle_loop(walls, front_x: float, drop_x: list[float]) -> _Loop:
    """The loop round the walls, its wake points provisional (evenly down the right side's middle)."""
    pts = np.concatenate([np.stack([w.element.airfoil.x, w.element.airfoil.y], 1) for w in walls])
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
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(chain, axis=0).T))])
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


def _distribute_loop(curves: dict[str, np.ndarray], counts: dict[str, int], fixed: dict[str, np.ndarray]):
    """Middle-loop edges not ``fixed``: Vinokur, the spacing at each corner
    shared by the edges meeting there (the geometric mean of their means)."""
    ends = {e: (tuple(np.round(c[0], 12)), tuple(np.round(c[-1], 12))) for e, c in curves.items()}
    length = {e: float(np.sum(np.hypot(*np.diff(c, axis=0).T))) for e, c in curves.items()}
    mean = {e: length[e] / counts[e] for e in curves}
    corner = {}
    for e, (p, q) in ends.items():
        for v in (p, q):
            corner.setdefault(v, []).append(mean[e])
    ds = {v: float(np.exp(np.mean(np.log(m)))) for v, m in corner.items()}
    out = {}
    for e, c in curves.items():
        if e in fixed:
            out[e] = fixed[e]
        else:
            out[e] = distribute(c, counts[e], ds[ends[e][0]], ds[ends[e][1]])
    return out


def _off_wall(blocks, nodes, layers, walls, h_ml) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per fill block, the cell heights wanted off its bottom and top sides:
    a layer's next step on a layer contour, the middle loop's spacing on the
    loop, and along a cut the geometric interpolation of its two ends' (the
    same for the blocks either side of it)."""
    off = {}
    ends = {}
    for k, (wall, layer) in enumerate(zip(walls, layers)):
        steps = layer.next_steps()
        n = len(layer.nodes) - 1
        for a, b, _, _ in wall.pieces():
            i0, i1 = layer.index[a], (layer.index[b] if layer.index[b] > layer.index[a] else n)
            off[_edge(k, a, b)] = steps[i0:i1 + 1]
            ends[tuple(np.round(layer.outer(i0), 12))] = steps[i0]
            ends[tuple(np.round(layer.outer(i1), 12))] = steps[i1]
    for e, c in nodes.items():
        if e.startswith("ml_"):
            off[e] = np.full(len(c), h_ml)
        elif e not in off:                                   # a cut
            h0, h1 = (ends.get(tuple(np.round(c[k], 12)), h_ml) for k in (0, -1))
            u = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
            off[e] = h0 * (h1 / h0) ** (u / u[-1])
    return {b.name: (chain_values(b.bottom, off), chain_values(b.top, off)) for b in blocks}


def multi_element(elements: list[Element], settings: Settings) -> tuple[list[RawBlock], dict]:
    """The grid's blocks (layers, fill blocks, outer ring) and a record of
    what was built (family, counts, cells).

    Raises:
        ValueError: a configuration the template does not take, or a grid
        that folds.
    """
    name = family([e.role for e in elements])
    walls = _walls(elements)
    blocks = graph(walls)
    last = len(walls) - 1
    pts = np.concatenate([np.stack([w.element.airfoil.x, w.element.airfoil.y], 1) for w in walls])
    size = MIDDLE_LOOP * float(np.ptp(pts[:, 0]))
    edges = set().union(*(b.edges for b in blocks))
    front_start = "b" if "b" in walls[0].landmarks else "lip"
    drops = [k for k in range(len(walls)) if f"drop{k}" in edges]

    def loop_through(point):
        """The middle loop placed by the vertices (``point(k, name)``)."""
        front_x = 0.5 * (point(0, "a")[0] + point(0, front_start)[0])
        return _middle_loop(walls, front_x, [float(point(k, "lip")[0]) for k in drops])

    # Counts first, from the walls (the loop placed by the wall landmarks).
    on_wall = lambda k, v: walls[k].points(np.array([walls[k].landmarks[v]]))[0]  # noqa: E731
    requests = _requests(walls, settings, loop_through(on_wall))
    counts = solve_counts(blocks, {e: n for e, n in requests.items() if e in edges}, settings.even_cells)

    layers = _layers(walls, counts, settings)
    nodes: dict[str, np.ndarray] = {}
    for k, (wall, layer) in enumerate(zip(walls, layers)):
        n = len(layer.nodes) - 1
        for a, b, _, _ in wall.pieces():
            i0, i1 = layer.index[a], (layer.index[b] if layer.index[b] > layer.index[a] else n)
            nodes[_edge(k, a, b)] = layer.nodes[i0:i1 + 1, -1]
    vertex = lambda k, v: layers[k].index[v]  # noqa: E731
    lips = {k: layers[k].outer(vertex(k, "lip")) for k, w in enumerate(walls) if "lip" in w.landmarks}
    loop = loop_through(lambda k, v: layers[k].outer(vertex(k, v)))
    h_ml = float(np.mean([np.sum(np.hypot(*np.diff(c, axis=0).T)) / counts[e]
                          for e, c in loop.curves().items() if not e.startswith("ml_right")]))

    def cut(k0, v0, k1, v1, n_cells, end=None, end_dir=None, end_step=None, lean=None):
        """A connector from layer ``k0``'s vertex ``v0`` along its outward
        direction (turned half way to ``lean``, if given) to layer ``k1``'s
        vertex ``v1`` (or to ``end``, arriving along ``end_dir``)."""
        i0 = vertex(k0, v0)
        start, out0, ds0 = layers[k0].outer(i0), layers[k0].outward(i0), layers[k0].step(i0)
        if lean is not None:
            out0 = out0 + lean / np.hypot(*lean)
        if end is None:
            i1 = vertex(k1, v1)
            end, end_dir, end_step = layers[k1].outer(i1), -layers[k1].outward(i1), layers[k1].step(i1)
        return distribute(hermite(start, out0, end, end_dir), n_cells, ds0, end_step)

    down = np.array([0.0, -1.0])
    for k in range(1, len(walls)):
        foot = layers[k].outer(vertex(k, "m"))
        nodes[f"gap{k}"] = cut(k - 1, _gap_start(walls, k - 1), k, "g", counts[f"gap{k}"])
        # the mouth and the drop leave a lip together, leaning apart
        nodes[f"mouth{k}"] = cut(k - 1, "lip", k, "m", counts[f"mouth{k}"], lean=foot - lips[k - 1])
    for foot, k in zip(loop.feet, drops):
        nodes[f"drop{k}"] = cut(k, "lip", None, None, counts[f"drop{k}"], foot, down, h_ml, lean=down)
    nodes["cut_a"] = cut(0, "a", None, None, counts["cut_a"], loop.qa, np.array([0.0, 1.0]), h_ml)
    nodes["cut_b"] = cut(0, front_start, None, None, counts["cut_b"], loop.qb, down, h_ml,
                         lean=None if front_start == "b" else down)
    # Wakes, back to front: the last from its trailing edge to the loop's
    # right side; each other over the next slot's chain.
    tip = layers[last].outer(vertex(last, "tip"))
    y_end = float(np.clip(tip[1], loop.centre[1] - 0.5 * size, loop.centre[1] + 0.5 * size))
    nodes[f"wake{last}"] = cut(last, "tip", None, None, counts[f"wake{last}"], np.array([loop.rt[0], y_end]),
                               np.array([1.0, 0.0]), h_ml)
    slots = {b.name: b for b in blocks}
    for k in range(last - 1, -1, -1):
        chain = chain_nodes(slots[f"slot{k + 1}"].bottom, nodes)
        nodes[f"wake{k}"] = _offset(chain, layers[k].outer(vertex(k, _wake_start(walls, k))), loop.rt[0])
    loop = loop.with_wakes([nodes[f"wake{k}"][-1] for k in range(len(walls))])
    if not loop.rt[1] > loop.wakes[0][1]:
        raise ValueError("the wakes reach above the middle loop: outside the template's envelope")
    nodes.update(_distribute_loop(loop.curves(), counts, {}))

    grids = smooth(fill(blocks, nodes), _off_wall(blocks, nodes, layers, walls, h_ml))

    # The outer ring: the middle loop, clockwise from the top front corner, marched out.
    right = [nodes[f"ml_right{k}"][::-1][:-1] for k in range(len(walls))]
    bottom = [nodes[f"ml_bottom{k}"][::-1][:-1] for k in reversed(range(len(loop.feet) + 1))]
    ring_line = np.concatenate([nodes["ml_top"][:-1], *right, nodes["ml_right_lower"][:-1], *bottom,
                                nodes["ml_front"][:-1]])
    reach = settings.far_field - size
    ring_cells = _even(cells_for(reach, h_ml, reach, 1.2), settings.even_cells)
    ring = march_to(ring_line, pts, ring_cells, h_ml, settings.far_field, periodic=True)
    ring = np.concatenate([ring, ring[:1]], axis=0)

    raw = [RawBlock(layer.nodes[..., 0].copy(), layer.nodes[..., 1].copy(), name=f"layer{k}")
           for k, layer in enumerate(layers)]
    raw += raw_blocks(grids)
    raw.append(RawBlock(ring[..., 0].copy(), ring[..., 1].copy(), name="ring"))
    record = {"family": name, "blocks": len(raw), "cells": int(sum((b.x.shape[0] - 1) * (b.x.shape[1] - 1)
                                                                    for b in raw)),
              "ring_cells": ring_cells, "middle_loop_spacing": h_ml}
    return raw, record
