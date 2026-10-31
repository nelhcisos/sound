"""Сцена «линия из мха»: скала из столбов, средние камни, валуны + точка обзора и узор.

Все размеры — мм итоговой модели, верх основания (вода) — z = 0.
"""
import numpy as np
from anamorph import Convex, Camera, overlap, unit, catmull_rom, snap
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

SEED = 11
ZB = -2.0  # низ камней при генерации (ниже воды всё равно станет штырём)

# Точка обзора (глаз) и куда смотрим
EYE = np.array([-6.0, -172.0, 62.0])
LOOK = np.array([0.0, 52.0, 20.0])

# Узор задаётся в плоскости фронта скалы y = Y_REF: кружок-старт справа и волнистая линия
Y_REF = 54.0
CIRCLE = (53.0, 19.5)
CIRCLE_R = 6.0
LINE_PTS = [(53, 19.5), (43, 17), (33, 19.5), (22, 23), (11, 21), (0, 16.5),
            (-11, 15.5), (-22, 19), (-33, 22.5), (-44, 20.5), (-55, 16.5), (-66, 18)]
HALF_W = 2.5

# Ключевые камни на пути линии: x на скале, глубина y, полуоси, на сколько верх выше луча
KEY_ROCKS = [
    (47.0, 36.0, (8.5, 6.5, 0), 1.5),
    (27.0, 17.0, (7.0, 6.0, 0), 3.0),
    (6.0, 41.0, (9.0, 6.0, 0), 4.5),
    (-15.0, 27.0, (8.0, 6.0, 0), 0.5),
    (-37.0, 15.0, (7.0, 6.0, 0), 3.5),
    (-60.0, 38.0, (8.0, 6.5, 0), 2.0),
]


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def chisel(rng, s, n_front=(3, 6), n_top=(1, 3), front_depth=(1.0, 4.5), top_depth=(1.0, 4.5)):
    """Сколы: косые передние грани, V-щели между столбами, скосы верхних рёбер."""
    for _ in range(rng.integers(*n_front)):
        n = unit([rng.uniform(-1.3, 1.3), -1.0, rng.uniform(-0.45, 0.9)])
        s = s.add([*n, s.support(n) - rng.uniform(*front_depth)]).cleaned()
    for _ in range(rng.integers(*n_top)):
        a = rng.uniform(0, 2 * np.pi)
        k = rng.uniform(0.7, 2.2)
        n = unit([k * np.cos(a), k * np.sin(a), 1.0])
        s = s.add([*n, s.support(n) - rng.uniform(*top_depth)]).cleaned()
    return s


def cliff(rng, x0=-84.0, x1=84.0, y0=52.0, y1=80.0, n_front=12, n_back=10,
          block_edges=(-50.0, -16.0, 18.0, 52.0), gap_between=0.12):
    seeds = []
    for row, n, ymid in (("front", n_front, y0 + 6.5), ("back", n_back, y1 - 7.5)):
        xs = np.linspace(x0, x1, n + 1)
        cx = (xs[:-1] + xs[1:]) / 2 + rng.uniform(-2.5, 2.5, n)
        for x in cx:
            seeds.append((x, ymid + rng.uniform(-2.5, 2.5), row))
    S2 = np.array([[a, b] for a, b, _ in seeds])
    block = np.searchsorted(np.array(block_edges), S2[:, 0])
    stones = []
    for i, (sx, sy, row) in enumerate(seeds):
        P = [[-1, 0, 0, -x0], [1, 0, 0, x1], [0, 1, 0, y1], [0, -1, 0, -(y0 - 9)], [0, 0, -1, -ZB]]
        for j in range(len(seeds)):
            if j == i:
                continue
            dvec = S2[j] - S2[i]
            L = np.linalg.norm(dvec)
            n = np.array([dvec[0] / L, dvec[1] / L, 0.0])
            gap = 0.0 if block[i] == block[j] else gap_between
            P.append([*n, n[:2] @ (S2[i] + S2[j]) / 2 - gap / 2])
        prof = 1 - 0.4 * smooth((abs(sx) - 56) / 28)
        H = (rng.uniform(27, 43) if row == "front" else rng.uniform(42, 60)) * prof
        tx, ty = rng.uniform(-0.6, 0.6, 2)
        nt = unit([tx, ty, 1.0])
        P.append([*nt, nt @ np.array([sx, sy, H])])
        if row == "front":  # свой неровный фронт у каждого столба
            nf = unit([rng.uniform(-0.6, 0.6), -1.0, rng.uniform(-0.2, 0.35)])
            P.append([*nf, nf @ np.array([sx, y0 + rng.uniform(-6.0, 2.5), 0.0])])
        s = Convex(P, kind="cliff", row=row, block=int(block[i])).cleaned()
        s = chisel(rng, s)
        stones.append(s)
    return stones


def rock(rng, c, axes, yaw, tilt=0.18, n_planes=18, rho=(0.86, 1.0), big=2):
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ax, ay = rng.uniform(-tilt, tilt, 2)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R = Rz @ Rx @ Ry
    D = np.diag(axes)
    k = np.arange(n_planes) + 0.5
    phi = np.arccos(1 - 2 * k / n_planes)
    th = np.pi * (1 + 5 ** 0.5) * k
    N = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    N = unit(N @ Q.T + rng.normal(scale=0.12, size=N.shape))
    P = []
    for i, n in enumerate(N):
        h = np.linalg.norm(D @ R.T @ n)
        r = rng.uniform(0.7, 0.8) if i < big else rng.uniform(*rho)
        P.append([*n, n @ c + r * h])
    P.append([0, 0, -1, -ZB])
    return Convex(P).cleaned()


def ray_at(x_ref, z_ref, y):
    """Точка луча «глаз → (x_ref, Y_REF, z_ref)» на глубине y."""
    Q = np.array([x_ref, Y_REF, z_ref])
    t = (y - EYE[1]) / (Q[1] - EYE[1])
    return EYE + t * (Q - EYE)


def line_z(x_ref):
    pts = np.array(LINE_PTS, float)
    o = np.argsort(pts[:, 0])
    return float(np.interp(x_ref, pts[o, 0], pts[o, 1]))


def footprint(s):
    """Площадь сечения камня на уровне воды, мм²."""
    from shapely.geometry import MultiPoint
    V, _ = s.clip_z(0).add([0, 0, 1, 1e-4]).geometry()
    return MultiPoint(V[:, :2]).convex_hull.area if len(V) > 2 else 0.0


def place_key(rng, stones, gap=0.5, tries=200):
    out = []
    for xr, y, axes, above in KEY_ROCKS:
        for it in range(tries):
            jx, jy = rng.uniform(-2, 2) * (it > 0), rng.uniform(-2, 2) * (it > 0)
            p = ray_at(xr + jx, line_z(xr + jx), y + jy)
            ax = np.array(axes) * rng.uniform(0.92, 1.08, 3)
            top = p[2] + above
            ax[2] = top / 1.33  # стоячий камень: широкий низ, центр на 0.4 высоты
            s = rock(rng, np.array([p[0], p[1], 0.4 * ax[2]]), ax, rng.uniform(0, np.pi), rho=(0.9, 1.0))
            V, _ = s.geometry()
            if len(V) == 0:
                continue
            s = s.add([0, 0, 1, top]) if V[:, 2].max() > top + 0.5 else s  # срезать слишком высокий верх
            s = s.cleaned()
            s.meta.update(kind="key")
            if footprint(s) < 90:
                continue
            if any(overlap(s, o, gap) for o in stones + out):
                continue
            out.append(s)
            break
        else:
            print("ключевой камень не встал:", xr, y)
    return out


def place(rng, stones, n, zone, axes_rng, cz_frac, kind, gap=0.5, tries=400):
    out = []
    for _ in range(n):
        for _ in range(tries):
            x = rng.uniform(*zone[0])
            y = rng.uniform(*zone[1])
            ax = np.array([rng.uniform(*axes_rng[0]), rng.uniform(*axes_rng[1]), rng.uniform(*axes_rng[2])])
            cz = ax[2] * rng.uniform(*cz_frac)
            s = rock(rng, np.array([x, y, cz]), ax, rng.uniform(0, np.pi))
            s.meta.update(kind=kind)
            V, _ = s.geometry()
            if len(V) == 0 or V[:, 2].max() < 2.0:
                continue
            if any(overlap(s, o, gap) for o in stones + out):
                continue
            out.append(s)
            break
    return out


def build_stones(seed=SEED):
    rng = np.random.default_rng(seed)
    stones = cliff(rng)
    stones += place_key(rng, stones)
    stones += place(rng, stones, 8, ((-76, 76), (26, 48)), ((5, 10), (4, 7), (6, 12)), (0.3, 0.8), "mid")
    stones += place(rng, stones, 12, ((-80, 80), (1, 25)), ((3.5, 11), (3, 7.5), (2.5, 7)), (-0.2, 0.5), "front")
    return stones


def camera():
    return Camera(EYE, LOOK)


def pattern(cam, n_per=120):
    pts = np.array([[x, Y_REF, z] for x, z in LINE_PTS])
    uv, dep = cam.project(pts)
    curve = catmull_rom(uv, n_per=n_per)
    w = HALF_W / float(dep.mean())
    line = LineString(curve).buffer(w, quad_segs=48)
    cuv, cdep = cam.project(np.array([[CIRCLE[0], Y_REF, CIRCLE[1]]]))
    circ = Point(cuv[0]).buffer(CIRCLE_R / float(cdep[0]), quad_segs=96)
    return snap(unary_union([line, circ])), curve, w
