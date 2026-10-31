#!/usr/bin/env python3
"""Анаморфная диорама «волосы сирены»: столбы, скала, статуя.

Красные слои = пересечение камней с «конусом волос» — лучами из точки взгляда
через контур волос. Из этой точки все красные куски складываются в одну прядь,
с любой другой — распадаются (параллакс между столбами и скалой).

Мм, Z вверх, вода (верх основания) — z=0, смотрим из -Y в +Y.
"""
import json
import os
import sys

import numpy as np
import manifold3d as mf
import trimesh
import mapbox_earcut as earcut
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from skimage import measure

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
RAISE = 0.35          # на сколько красный слой выступает над камнем, мм

# ---------------------------------------------------------------- helpers
def to_man(tm):
    return mf.Manifold(mf.Mesh(vert_properties=np.ascontiguousarray(tm.vertices, np.float32),
                               tri_verts=np.ascontiguousarray(tm.faces, np.uint32)))


def to_tm(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3].astype(float),
                           np.asarray(mm.tri_verts).astype(np.int64), process=True)


def union(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    if not ms:
        return mf.Manifold()
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add)


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def sp(v, e):
    return np.sign(v) * np.abs(v) ** e


def unit_pts(n, exy, ez, jit=0.10, rjit=0.06):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i + rng.uniform(0, 6.3)
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    d += rng.normal(0, jit, d.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = np.stack([sp(d[:, 0], exy), sp(d[:, 1], exy), sp(d[:, 2], ez)], 1)
    return p * rng.uniform(1 - rjit, 1, (n, 1))


class Block:
    """Грань-камень: выпуклая оболочка случайных точек (low-poly)."""

    def __init__(self, c, r, yaw, pts):
        self.c, self.r, self.yaw, self.pts = np.array(c, float), np.array(r, float), yaw, pts

    def man(self, d=0.0):
        p = self.pts * (self.r + d)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return mf.Manifold.hull_points(p @ R.T + self.c)


# ---------------------------------------------------------------- точка взгляда
CAM = np.array([-6.0, -185.0, 24.0])
TGT = np.array([-6.0, 10.0, 27.0])
FOV_V = 17.0                      # вертикальный угол обзора для превью, град
ASPECT = 2.0
fwd = (TGT - CAM) / np.linalg.norm(TGT - CAM)
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)

# ---------------------------------------------------------------- статуя (SDF)
SEAT = np.array([21.5, -4.5, 0.0])     # z посчитаем по камню
BODY_YAW = np.radians(-38)             # корпус повернут вправо и к зрителю
HEAD_YAW = np.radians(18)              # голова смотрит ещё правее
STATUE_SCALE = 1.15


def sd_ell(p, c, r, yaw=0.0):
    q = p - c
    if yaw:
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        q = np.stack([cy * q[:, 0] - sy * q[:, 1], sy * q[:, 0] + cy * q[:, 1], q[:, 2]], 1)
    r = np.asarray(r, float)
    k0 = np.linalg.norm(q / r, axis=1)
    k1 = np.linalg.norm(q / (r * r), axis=1)
    return k0 * (k0 - 1.0) / (k1 + 1e-12)


def sd_rcone(p, a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    l2 = ba @ ba
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - y[:, None] * ba
    x2 = np.einsum("ij,ij->i", xv, xv)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    r1 = np.sqrt(x2 + z2) * il2 - rb
    r2 = np.sqrt(x2 + y2) * il2 - ra
    r3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - ra
    return np.where(np.sign(z) * a2 * z2 > k, r1, np.where(np.sign(y) * a2 * y2 < k, r2, r3))


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def limb(p, pts, rads):
    d = None
    for i in range(len(pts) - 1):
        di = sd_rcone(p, pts[i], pts[i + 1], rads[i], rads[i + 1])
        d = di if d is None else np.minimum(d, di)
    return d


def statue_sdf(p):
    """Сидящая фигура в длинной драпировке. Локальные оси: x — куда смотрит
    корпус, y — левый бок, z — вверх; (0,0,0) — точка сиденья."""
    hy = HEAD_YAW
    d = sd_ell(p, (0.2, 0, 1.7), (2.0, 2.75, 1.8))                       # таз
    d = smin(d, sd_ell(p, (0.05, 0, 4.1), (1.55, 2.05, 2.0)), 1.2)       # живот
    d = smin(d, sd_ell(p, (-0.15, 0, 6.6), (1.75, 2.45, 2.1)), 1.2)      # грудная клетка
    d = smin(d, sd_ell(p, (0.55, 0, 6.35), (1.35, 2.1, 1.35)), 0.9)      # драпировка на груди
    d = smin(d, sd_rcone(p, (0.9, -1.9, 7.9), (1.3, 1.2, 6.2), 0.32, 0.3), 0.3)  # край ткани наискось
    d = smin(d, sd_rcone(p, (0.2, -2.3, 3.4), (0.35, 2.3, 3.4), 0.45, 0.45), 0.5)  # пояс
    d = smin(d, sd_rcone(p, (-0.3, 2.35, 8.15), (-0.3, -2.35, 8.15), 0.85, 0.85), 1.0)  # плечи
    d = smin(d, sd_rcone(p, (-0.25, 0, 8.4), (0.05, 0, 10.4), 0.72, 0.62), 0.5)        # шея
    d = smin(d, sd_ell(p, (0.15, 0, 11.75), (1.55, 1.32, 1.82), hy), 0.5)   # голова
    c, s = np.cos(hy), np.sin(hy)
    nose = np.array([0.15 + 1.55 * c, 1.55 * s, 11.7])
    d = smin(d, sd_ell(p, nose, (0.34, 0.24, 0.42), hy), 0.25)               # нос
    chin = np.array([0.15 + 1.1 * c, 1.1 * s, 10.55])
    d = smin(d, sd_ell(p, chin, (0.6, 0.6, 0.55), hy), 0.4)                  # подбородок
    # волосы: пучок за затылком, развевается назад (дальше его обрежет конус)
    d = smin(d, sd_rcone(p, (-0.6, 0.15, 12.1), (-3.1, 0.7, 12.45), 1.75, 1.3), 0.6)
    d = smin(d, sd_rcone(p, (-3.1, 0.7, 12.45), (-4.6, 1.0, 12.2), 1.3, 0.75), 0.4)
    # правая рука (к зрителю) — ладонью на камень
    d = smin(d, limb(p, [(-0.35, -2.55, 8.0), (-0.9, -3.45, 5.0), (-0.6, -3.95, 1.9), (-0.3, -4.1, 0.25)],
                     [0.62, 0.5, 0.4, 0.42]), 0.45)
    # левая рука — на колене
    d = smin(d, limb(p, [(-0.35, 2.55, 8.0), (0.7, 3.05, 5.25), (3.0, 1.6, 3.7), (4.2, 0.6, 3.6)],
                     [0.62, 0.5, 0.4, 0.42]), 0.45)
    # ноги: левая закинута на правую
    legR = limb(p, [(0.6, -1.25, 1.35), (5.3, -1.45, 1.5), (5.9, -1.55, -4.6), (6.0, -1.6, -5.4)],
                [1.25, 0.95, 0.62, 0.5])
    legL = limb(p, [(0.6, 1.25, 1.35), (5.1, -0.1, 3.35), (6.9, -2.6, -1.7), (7.0, -2.8, -2.6)],
                [1.25, 0.95, 0.62, 0.5])
    d = smin(d, np.minimum(legR, legL), 0.8)
    # драпировка (длинная юбка): от пояса через колени вниз
    skirt = limb(p, [(0.25, 0, 2.6), (5.0, -0.7, 2.4), (6.0, -1.6, -2.4), (6.2, -1.9, -3.4)],
                 [2.55, 1.85, 1.35, 1.25])
    d = smin(d, skirt, 0.9)
    # птичьи лапы (сирена): по три когтя вперёд
    for ank, yaw in (((6.0, -1.6, -5.4), -0.15), ((7.0, -2.8, -2.6), -0.35)):
        a = np.array(ank)
        for k, ang in enumerate((-0.45, 0.0, 0.45)):
            dirv = np.array([np.cos(yaw + ang), np.sin(yaw + ang), -0.55])
            tip = a + dirv * 1.9
            d = smin(d, sd_rcone(p, a, tip, 0.36, 0.17), 0.25)
        d = smin(d, sd_rcone(p, a, a + np.array([-1.0, 0, -0.5]), 0.3, 0.15), 0.2)
    return d


def build_statue(origin):
    h = 0.09
    lo = np.array([-8.5, -6.5, -8.5])
    hi = np.array([9.5, 6.5, 15.0])
    n = np.ceil((hi - lo) / h).astype(int) + 1
    gx, gy, gz = [lo[i] + h * np.arange(n[i]) for i in range(3)]
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    vol = statue_sdf(P).reshape(n)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(h, h, h))
    v = (v + lo) * STATUE_SCALE
    tm = trimesh.Trimesh(v, f[:, ::-1], process=True)
    tm = max(tm.split(only_watertight=True), key=lambda m: m.volume)
    if tm.volume < 0:
        tm.invert()
    c, s = np.cos(BODY_YAW), np.sin(BODY_YAW)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tm.vertices = tm.vertices @ R.T + origin
    return to_man(tm)


# ---------------------------------------------------------------- камни
ICO = trimesh.creation.icosphere(subdivisions=1).vertices      # для выступа красного слоя


class Convex:
    """Выпуклый гранёный кусок камня; man(d) — он же, раздутый на d мм."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, float)

    def man(self, d=0.0):
        p = self.pts if d == 0 else (self.pts[:, None, :] + d * ICO[None]).reshape(-1, 3)
        return mf.Manifold.hull_points(p)


def ring(n, r, z, phase, jit=0.07, ajit=0.22):
    a = phase + (np.arange(n) + rng.uniform(-ajit, ajit, n)) * 2 * np.pi / n
    rr = r * rng.uniform(1 - jit, 1 + jit, n)
    return np.stack([rr * np.cos(a), rr * np.sin(a), np.full(n, z)], 1)


def place(p, c, yaw, sx=1.0, sy=1.0):
    p = p * [sx, sy, 1.0]
    cy, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[cy, -s, 0], [s, cy, 0], [0, 0, 1]]).T + [c[0], c[1], 0]


def slab(c, r, z0, z1, n=None, bevel=0.24, inset=0.84, sx=1.0, sy=1.0, dome=False):
    """Плита столба: многогранная «таблетка» с фасками сверху и снизу."""
    n = n or int(rng.integers(7, 10))
    ph = rng.uniform(0, 2 * np.pi)
    h = z1 - z0
    if dome:
        pts = np.concatenate([ring(n, r * inset, z0, ph + np.pi / n),
                              ring(n, r, z0 + 0.22 * h, ph),
                              ring(n, r * 0.93, z0 + 0.5 * h, ph + np.pi / n),
                              ring(n - 2, r * 0.68, z0 + 0.8 * h, ph),
                              ring(4, r * 0.3, z1, ph + 0.4, jit=0.2)])
    else:
        b = bevel * h
        pts = np.concatenate([ring(n, r * inset, z0, ph + np.pi / n),
                              ring(n, r, z0 + b, ph),
                              ring(n, r * rng.uniform(0.95, 1.05), z1 - b, ph),
                              ring(n, r * inset, z1, ph + np.pi / n)])
    return Convex(place(pts, c, rng.uniform(0, 2 * np.pi), sx, sy))


def make_pillar(x, y, h, r, taper=0.14, lean=(0, 0), squash=(1, 1), lay=(5.5, 9.5)):
    """Столб из стопки плит разной ширины (горизонтальные борозды между ними)."""
    out, z = [], -0.8
    hs = []
    while sum(hs) < h + 0.8 - lay[0] * 1.3:
        hs.append(rng.uniform(*lay))
    hs.append(h + 0.8 - sum(hs))
    for i, hi in enumerate(hs):
        t = (z + hi / 2) / h
        top = i == len(hs) - 1
        ri = r * (1 - taper * t) * rng.uniform(0.88, 1.08)
        c = (x + lean[0] * t * h + rng.normal(0, 0.05 * r), y + lean[1] * t * h + rng.normal(0, 0.05 * r))
        out.append(slab(c, ri, z - (0.35 if i else 0), z + hi, dome=top,
                        sx=squash[0] * rng.uniform(0.9, 1.1), sy=squash[1] * rng.uniform(0.9, 1.1)))
        z += hi
    return out


def front_y(x):
    return 31.0 - 0.0045 * x * x


def make_cliff(height=38.0, depth=9.0):
    """Скала-задник: длинные пласты разной толщины, каждый со своей фаской и выносом."""
    out = []
    z0 = 0.0
    th = []
    while sum(th) < height - 5:
        th.append(rng.uniform(3.0, 8.5))
    th.append(height - sum(th))
    xs = np.arange(-66, 67, 7.0)
    for li, t in enumerate(th):
        z1 = z0 + t
        rec = rng.uniform(-2.2, 2.2) - (3.5 if li == 0 else 1.6 if li == 1 else 0)  # низ — уступ-берег
        k = rng.uniform(0.04, 0.09)
        ph = rng.uniform(0, 6.3)
        bb, bt = rng.uniform(0.5, 1.6), rng.uniform(0.5, 1.6)
        prof = []
        for x in xs:
            x = x + rng.uniform(-1.2, 1.2)
            fy = front_y(x) + rec + 1.3 * np.sin(k * x + ph) + rng.uniform(-0.45, 0.45)
            yaw = np.arctan(-2 * 0.0045 * x)
            tng = np.array([np.cos(yaw), np.sin(yaw)])
            nrm = np.array([-np.sin(yaw), np.cos(yaw)])
            base = np.array([x, front_y(x)])
            f = base + nrm * (fy - front_y(x))
            zj = lambda: rng.uniform(-0.25, 0.25)
            P = [(f + nrm * bb, z0), (f + nrm * rng.uniform(-0.3, 0.3), z0 + 0.32 * t + zj()),
                 (f + nrm * rng.uniform(-0.3, 0.3), z1 - 0.32 * t + zj()), (f + nrm * bt, z1 + zj() * 0.5),
                 (base + nrm * (depth + 4), z1), (base + nrm * (depth + 4), z0)]
            prof.append(np.array([[p[0][0], p[0][1], p[1]] for p in P]))
        for a, b in zip(prof[:-1], prof[1:]):
            step = (b[:, :2] - a[:, :2]).mean(0)
            step = np.append(step / np.linalg.norm(step) * 0.15, 0.0)
            out.append(Convex(np.concatenate([a, b + step])))
        z0 = z1 - 0.3
    return out


def boulder(c, r, e=0.7, n=22):
    p = unit_pts(n, 0.9, e) * r
    return Convex(place(p, c, rng.uniform(0, 2 * np.pi)) + [0, 0, c[2]])


# ---------------------------------------------------------------- волосы (2D, плоскость головы)
def hair_polygon():
    # центральная линия пряди: u — вправо (к лицу), v — вверх; 0,0 — центр головы
    cu = np.array([-0.6, -4.5, -10, -17, -24, -31, -38, -45, -52, -59, -66])
    cv = np.array([0.5, 1.35, 1.0, -0.6, -1.9, -1.4, 0.4, 1.7, 1.3, -0.2, -1.3])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cu), np.diff(cv)))])
    fu, fv = CubicSpline(s, cu), CubicSpline(s, cv)
    L = s[-1]

    def ribbon(s0, s1, off_fn, w_fn, n=120):
        ss = np.linspace(s0, s1, n)
        pu, pv = fu(ss), fv(ss)
        du, dv = fu(ss, 1), fv(ss, 1)
        nn = np.hypot(du, dv)
        nu, nv = -dv / nn, du / nn
        o, w = off_fn(ss), np.maximum(w_fn(ss), 0.0)
        top = np.stack([pu + nu * (o + w), pv + nv * (o + w)], 1)
        bot = np.stack([pu + nu * (o - w), pv + nv * (o - w)], 1)
        return Polygon(np.concatenate([top, bot[::-1]]))

    def wmain(ss):
        return np.interp(ss, [0, 3, 9, 18, 30, 42, 54, L], [2.3, 2.75, 2.95, 2.8, 2.35, 1.75, 1.05, 0.35])

    parts = [ribbon(0, 38, lambda ss: 0 * ss, wmain)]
    # три пряди к кончику
    for k, (endf, spread, lenf) in enumerate(((1, 0.9, 0.86), (0, 0.0, 1.0), (-1, -0.8, 0.93))):
        s0, s1 = 34.0, L * lenf

        def off(ss, endf=endf, spread=spread, s0=s0):
            t = np.clip((ss - s0) / (L - s0), 0, 1)
            return endf * wmain(ss) * 0.62 + spread * t ** 1.5 * 1.6

        def ww(ss, s1=s1, s0=s0):
            t = np.clip((ss - s0) / (s1 - s0), 0, 1)
            return np.maximum(wmain(ss) * 0.46 * (1 - t ** 1.4), 0.13)   # кончик не тоньше ~0.3 мм

        parts.append(ribbon(s0, s1, off, ww))
    # корни — шапка волос на затылке и макушке, лицо открыто
    th = np.linspace(0, 2 * np.pi, 120)
    S = STATUE_SCALE
    cap = Polygon(np.stack([S * (-0.25 + 2.08 * np.cos(th)), S * (0.22 + 2.2 * np.sin(th))], 1))
    hairline = Polygon([(-0.55 * S, -3.5 * S), (1.05 * S, 3.5 * S), (-8, 3.5 * S), (-8, -3.5 * S)])
    parts.append(cap.intersection(hairline))
    poly = unary_union(parts).buffer(0.05).buffer(-0.05).buffer(0.07, quad_segs=4).simplify(0.01)   # кончики скруглены
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = Polygon(poly.exterior)          # без дыр
    return poly


def make_cone(poly, H, s_near=0.55, s_far=1.6):
    xy = np.asarray(poly.exterior.coords)[:-1]
    if Polygon(xy).exterior.is_ccw is False:
        xy = xy[::-1]
    Q = H + xy[:, :1] * right + xy[:, 1:2] * up
    D = Q - CAM
    near = CAM + s_near * D
    far = CAM + s_far * D
    n = len(xy)
    tri = earcut.triangulate_float64(xy, np.array([n], np.uint32)).reshape(-1, 3)
    V = np.concatenate([near, far])
    F = [tri[:, ::-1], tri + n]
    i = np.arange(n)
    j = (i + 1) % n
    F.append(np.stack([i, j, j + n], 1))
    F.append(np.stack([i, j + n, i + n], 1))
    tm = trimesh.Trimesh(V, np.concatenate(F), process=True)
    trimesh.repair.fix_normals(tm)
    assert tm.is_watertight, "cone not watertight"
    return to_man(tm), Q


# ================================================================= сцена
print("камни…")
X0, X1, Y0, Y1 = -56.0, 56.0, -42.0, 42.0
footprint = box(X0 + 0.5, X1 - 0.5, Y0 + 0.5, Y1 - 0.5, -0.6, 80)
footprint_dil = box(X0 + 0.15, X1 - 0.15, Y0 + 0.15, Y1 - 0.15, -0.55, 80)   # красный выступ и на срезе

pillar_specs = {
    "p3_behind": dict(x=11.0, y=8.0, h=55, r=6.4, lean=(0.02, 0.0)),
    "p2_twin": dict(x=-4.0, y=15.0, h=56, r=6.0),
    "p1_twin": dict(x=-15.5, y=13.0, h=49, r=5.7, lean=(-0.03, 0.0)),
    "p4_back": dict(x=-33.0, y=17.0, h=24, r=5.4),
    "p6_right": dict(x=30.0, y=17.0, h=36, r=5.3),
    "p5_slab": dict(x=44.0, y=-9.0, h=43, r=6.2, squash=(1.3, 0.72)),
    "f1_front": dict(x=-37.0, y=-22.0, h=15, r=5.3),
    "f2_front": dict(x=-23.0, y=-10.0, h=21, r=5.6),
}
groups = {}
for k, sp_ in pillar_specs.items():
    groups[k] = make_pillar(**sp_)
groups["cliff"] = make_cliff()
# груда плит под статуей
groups["seat"] = [
    slab((SEAT[0] - 1.0, SEAT[1] + 1.5), 8.6, -0.8, 5.0, bevel=0.3, inset=0.8, sx=1.15, sy=0.9),
    slab((SEAT[0] - 0.2, SEAT[1] + 1.8), 7.0, 4.6, 10.0, bevel=0.3, inset=0.8, sx=1.05, sy=0.92),
    slab((SEAT[0] - 2.6, SEAT[1] + 2.4), 6.0, 9.6, 15.2, bevel=0.3, inset=0.8, sx=1.0, sy=0.95),
    slab((SEAT[0] + 6.5, SEAT[1] - 1.0), 4.0, 4.6, 8.4, bevel=0.3, inset=0.8),          # ступенька под ноги
]
groups["rubble"] = [
    boulder((-47, 13, 1.0), (4.2, 3.2, 2.0)), boulder((-26, 22, 0.8), (3.2, 2.6, 1.6)),
    boulder((35, 3, 0.8), (3.4, 2.8, 1.8)), boulder((7, -15, 0.5), (2.6, 2.2, 1.3)),
    boulder((-8, 3, 0.6), (2.8, 2.4, 1.4)), boulder((-30, -2, 0.5), (2.2, 1.9, 1.2)),
]

objs, objs_dil = {}, {}
for k, blocks in groups.items():
    objs[k] = union([b.man() for b in blocks]) ^ footprint
    objs_dil[k] = union([b.man(RAISE) for b in blocks]) ^ footprint_dil

# точка сиденья — верх верхнего камня груды
seat_top = objs["seat"].ray_cast([SEAT[0], SEAT[1], 40], [SEAT[0], SEAT[1], 0])
SEAT[2] = seat_top[0].position[2] if hasattr(seat_top[0], "position") else 16.5
print("сиденье z =", SEAT[2])
print("статуя…")
statue = build_statue(SEAT + np.array([0, 0, -0.25]))
HEAD = SEAT + np.array([0, 0, -0.25]) + STATUE_SCALE * np.array(
    [0.15 * np.cos(BODY_YAW), 0.15 * np.sin(BODY_YAW), 11.75])

print("конус волос…")
hp = hair_polygon()
cone, Q = make_cone(hp, HEAD)

print("булевы…")
stone, red = {}, {}
for k in objs:
    inside, outside = objs[k].split(cone)
    stone[k] = outside
    if not inside.is_empty():
        red[k] = objs_dil[k] ^ cone
st_red, st_stone = statue.split(cone)

R0 = 3.0
base_tm = trimesh.creation.box(extents=[X1 - X0, Y1 - Y0, 3.0])
base_tm.apply_translation([(X0 + X1) / 2, (Y0 + Y1) / 2, -1.5])
base = to_man(base_tm)

# ---------------------------------------------------------------- проверка: всё ли внутри контура красное
def first_hits(origin, dirs, meshes):
    """Для каждого луча — имя первого пересечённого объекта."""
    best_t = np.full(len(dirs), np.inf)
    best = np.array([""] * len(dirs), dtype=object)
    for name, tm in meshes.items():
        locs, ri, _ = tm.ray.intersects_location(np.repeat(origin[None], len(dirs), 0), dirs, multiple_hits=False)
        if len(ri) == 0:
            continue
        t = np.linalg.norm(locs - origin, axis=1)
        upd = t < best_t[ri]
        best_t[ri[upd]] = t[upd]
        best[ri[upd]] = name
    return best


mesh_red = {f"red:{k}": to_tm(v) for k, v in red.items()}
mesh_red["red:statue"] = to_tm(st_red)
mesh_other = {f"stone:{k}": to_tm(v) for k, v in stone.items()}
mesh_other["statue"] = to_tm(st_stone)
allm = {**mesh_red, **mesh_other}
minx, miny, maxx, maxy = hp.bounds
uu, vv = np.meshgrid(np.linspace(minx, maxx, 260), np.linspace(miny, maxy, 40))
from shapely import contains_xy
inside = contains_xy(hp.buffer(-0.12), uu.ravel(), vv.ravel())
pts2 = np.stack([uu.ravel()[inside], vv.ravel()[inside]], 1)
P3d = HEAD + pts2[:, :1] * right + pts2[:, 1:] * up
dirs = P3d - CAM
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
hits = first_hits(CAM, dirs, allm)
n_red = sum(h.startswith("red") for h in hits)
n_sky = sum(h == "" for h in hits)
bad = [h for h in hits if h and not h.startswith("red")]
print(f"лучей внутри волос: {len(hits)}, красных {n_red}, небо {n_sky}, не красных {len(bad)}",
      {b: bad.count(b) for b in set(bad)})
np.save(os.path.join(OUT, "hair_rays_bad.npy"), pts2[[not h.startswith("red") for h in hits]])

# ---------------------------------------------------------------- экспорт (итоговый размер)
WIDTH = float(os.environ.get("WIDTH", 100))      # ширина основания, мм
SC = WIDTH / (X1 - X0)
sc = lambda m: m.scale([SC, SC, SC])
CAMs, TGTs, HEADs, Qs = CAM * SC, TGT * SC, HEAD * SC, Q * SC
BASE_T = 3.0 * SC


def yup(tm):
    tm = tm.copy()
    v = tm.vertices.copy()
    tm.vertices = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    return tm


def sight_post():
    """Прицел: столбик с глазком Ø3 мм ровно в точке взгляда (ставится на стол)."""
    eye = CAMs.copy()
    hz = eye[2] + BASE_T                       # высота глазка над столом
    d = (TGTs - CAMs); d[2] = 0; d /= np.linalg.norm(d)
    yaw = np.arctan2(d[1], d[0]) - np.pi / 2
    foot = mf.Manifold.cylinder(2.0, 11, 11, 48).translate([0, 0, 0])
    col = mf.Manifold.cube([7, 4, hz - 4.5]).translate([-3.5, -2, 1.0])
    ring_ = (mf.Manifold.cylinder(4, 6.0, 6.0, 48, center=True)
             - mf.Manifold.cylinder(6, 1.5, 1.5, 32, center=True)).rotate([90, 0, 0]).translate([0, 0, hz])
    arrow = mf.Manifold.cube([2, 6, 0.6]).translate([-1, 5.5, 2.0])     # стрелка «к диораме»
    m = union([foot, col, ring_, arrow]).rotate([0, 0, np.degrees(yaw)])
    return m.translate([eye[0], eye[1], -BASE_T])


base_s = sc(base)
stone_s = {k: sc(v) for k, v in stone.items()}
red_s = {k: sc(v) for k, v in red.items()}
st_stone_s, st_red_s = sc(st_stone), sc(st_red)
post = sight_post()

# превью-сцена для рендера и веб-просмотра (статуя прорежена)
scene = trimesh.Scene()
scene.add_geometry(yup(to_tm(base_s)), node_name="base", geom_name="base")
for k, v in stone_s.items():
    scene.add_geometry(yup(to_tm(v)), node_name=f"stone_{k}", geom_name=f"stone_{k}")
for k, v in red_s.items():
    scene.add_geometry(yup(to_tm(v)), node_name=f"red_{k}", geom_name=f"red_{k}")
st_light = to_tm(st_stone_s)
try:
    import fast_simplification
    vv, ff = fast_simplification.simplify(st_light.vertices, st_light.faces, target_reduction=0.8)
    st_light = trimesh.Trimesh(vv, ff)
except Exception as e:
    print("decimate:", e)
scene.add_geometry(yup(st_light), node_name="statue", geom_name="statue")
scene.add_geometry(yup(to_tm(st_red_s)), node_name="red_statue", geom_name="red_statue")
scene.add_geometry(yup(to_tm(post)), node_name="post", geom_name="post")
scene.export(os.path.join(OUT, "scene.glb"))

y3 = lambda p: [float(p[0]), float(p[2]), float(-p[1])]
view = dict(cam=y3(CAMs), tgt=y3(TGTs), fov=FOV_V, aspect=ASPECT, head=y3(HEADs),
            hair=[y3(q) for q in Qs], scale=SC, base_t=BASE_T,
            bounds=dict(x0=X0 * SC, x1=X1 * SC, y0=Y0 * SC, y1=Y1 * SC))
json.dump(view, open(os.path.join(OUT, "view.json"), "w"))

red_all = union(list(red_s.values()) + [st_red_s])
whole = union([base_s] + [sc(v) for v in objs.values()] + [sc(statue)])
one = union([whole, red_all])
stone_all = whole - red_all


def check(name, m):
    t = to_tm(m)
    t2 = trimesh.Trimesh(t.vertices.astype(np.float32), t.faces, process=True)   # как после STL
    ok = t2.is_watertight and len(t2.split(only_watertight=False)) == len(m.decompose())
    print(f"  {name}: тел {len(m.decompose())}, замкнуто после STL: {t2.is_watertight}")
    return ok


for nm, mm in (("одно тело", one), ("камень", stone_all), ("красное", red_all)):
    check(nm, mm)
info = dict(
    width=WIDTH, depth=round((Y1 - Y0) * SC, 1), height=round(one.bounding_box()[5] + BASE_T, 1),
    volume_cm3=round(one.volume() / 1000, 1), red_cm3=round(red_all.volume() / 1000, 2),
    eye_from_front_edge=round(-(CAMs[1] - Y0 * SC), 1), eye_left_of_center=round(-CAMs[0], 1),
    eye_above_table=round(CAMs[2] + BASE_T, 1), eye_to_statue=round(float(np.linalg.norm(HEADs - CAMs)), 1),
)
print(info)
json.dump(info, open(os.path.join(OUT, "info.json"), "w"), ensure_ascii=False)
tag = f"{int(round(WIDTH))}mm"
to_tm(one).export(os.path.join(OUT, f"siren_rocks_{tag}.stl"))
to_tm(red_all).export(os.path.join(OUT, f"siren_rocks_{tag}_red_only.stl"))
to_tm(post).export(os.path.join(OUT, "sight_post.stl"))
s3 = trimesh.Scene()
s3.add_geometry(to_tm(stone_all), node_name="stone", geom_name="stone")
s3.add_geometry(to_tm(red_all), node_name="red", geom_name="red")
open(os.path.join(OUT, f"siren_rocks_{tag}_2color.3mf"), "wb").write(trimesh.exchange.threemf.export_3MF(s3))
print("готово")
