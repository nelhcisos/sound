"""Анаморфная проекция рисунка на набор выпуклых низкополигональных камней.

Идея: из одной точки обзора E рисунок (линия с кружком) проецируется на
видимые грани камней. Каждый камень — выпуклый многогранник (пересечение
полупространств), поэтому видимость считается точно, полигонами в плоскости
изображения, а мох получается чистыми кусками с ровными краями.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
import shapely
from shapely.geometry import Polygon, MultiPoint, LineString, Point
from shapely.ops import unary_union, polygonize

GRID = 1e-9  # сетка округления в координатах изображения (~0.0002 мм)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _dedupe(V, tol=1e-7):
    out = []
    for v in V:
        if not any(np.linalg.norm(u - v) < tol for u in out):
            out.append(v)
    return np.array(out)


class Convex:
    """Выпуклый многогранник: строки P = [n, d], n·x <= d, |n| = 1."""

    def __init__(self, planes, **meta):
        P = np.asarray(planes, float).reshape(-1, 4)
        n = np.linalg.norm(P[:, :3], axis=1, keepdims=True)
        self.P = P / n
        self.meta = dict(meta)
        self._geo = None

    def copy(self, planes=None, **meta):
        m = dict(self.meta)
        m.update(meta)
        return Convex(self.P if planes is None else planes, **m)

    def add(self, planes):
        return self.copy(np.vstack([self.P, np.asarray(planes, float).reshape(-1, 4)]))

    def grow(self, delta):
        P = self.P.copy()
        P[:, 3] += delta
        return self.copy(P)

    def clip_z(self, z0):
        return self.add([0, 0, -1, -z0])

    def interior(self):
        A = np.hstack([self.P[:, :3], np.ones((len(self.P), 1))])
        res = linprog([0, 0, 0, -1], A_ub=A, b_ub=self.P[:, 3],
                      bounds=[(None, None)] * 3 + [(0, None)], method="highs")
        if res.status != 0:
            return None, -1.0
        return res.x[:3], res.x[3]

    def geometry(self):
        """Вершины и грани (индекс плоскости, вершины по кругу против часовой со стороны нормали)."""
        if self._geo is not None:
            return self._geo
        x0, r = self.interior()
        if x0 is None or r < 1e-6:
            self._geo = (np.zeros((0, 3)), [])
            return self._geo
        hs = np.hstack([self.P[:, :3], -self.P[:, 3:4]])
        V = _dedupe(HalfspaceIntersection(hs, x0).intersections)
        facets = []
        for i in range(len(self.P)):
            n, d = self.P[i, :3], self.P[i, 3]
            on = np.where(np.abs(V @ n - d) < 1e-6)[0]
            if len(on) < 3:
                continue
            c = V[on].mean(0)
            e = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
            a = unit(np.cross(n, e))
            b = np.cross(n, a)
            ang = np.arctan2((V[on] - c) @ b, (V[on] - c) @ a)
            facets.append((i, on[np.argsort(ang)]))
        self._geo = (V, facets)
        return self._geo

    def facet_area(self, V, idx):
        p = V[idx]
        return 0.5 * np.linalg.norm(np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(0))

    def cleaned(self, min_area=0.05):
        """Убирает почти вырожденные плоскости (крошечные грани)."""
        cur = self
        for _ in range(6):
            V, facets = cur.geometry()
            keep = [i for i, idx in facets if cur.facet_area(V, idx) >= min_area]
            if len(keep) == len(facets) and len(keep) == len(cur.P):
                return cur
            cur = cur.copy(cur.P[keep])
        return cur

    def mesh(self):
        V, facets = self.geometry()
        F = []
        for _, idx in facets:
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
        return V, np.array(F, dtype=np.int64).reshape(-1, 3)

    def volume(self):
        V, F = self.mesh()
        if len(F) == 0:
            return 0.0
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

    def support(self, n):
        V, _ = self.geometry()
        return float((V @ np.asarray(n, float)).max())

    def ray_enter(self, E, D):
        """Параметр t входа луча E + t·D (None — мимо)."""
        nd = self.P[:, :3] @ D
        rhs = self.P[:, 3] - self.P[:, :3] @ E
        t0, t1 = -np.inf, np.inf
        pos, neg, zer = nd > 1e-15, nd < -1e-15, np.abs(nd) <= 1e-15
        if np.any(rhs[zer] < 0):
            return None
        if pos.any():
            t1 = np.min(rhs[pos] / nd[pos])
        if neg.any():
            t0 = np.max(rhs[neg] / nd[neg])
        return t0 if t0 <= t1 else None


def overlap(a, b, gap=0.0):
    """True, если многогранники ближе gap (раздутые на gap/2 пересекаются)."""
    P = np.vstack([a.P, b.P]).copy()
    P[:, 3] += gap / 2
    A = np.hstack([P[:, :3], np.ones((len(P), 1))])
    res = linprog([0, 0, 0, -1], A_ub=A, b_ub=P[:, 3],
                  bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    return res.status == 0 and res.x[3] > 1e-6


class Camera:
    def __init__(self, E, T):
        self.E = np.asarray(E, float)
        self.T = np.asarray(T, float)
        self.f = unit(self.T - self.E)
        self.r = unit(np.cross(self.f, [0, 0, 1.0]))
        self.u = np.cross(self.r, self.f)

    def project(self, X):
        c = np.atleast_2d(X) - self.E
        z = c @ self.f
        return np.stack([(c @ self.r) / z, (c @ self.u) / z], 1), z

    def rays(self, uv):
        uv = np.atleast_2d(uv)
        return self.f[None] + uv[:, :1] * self.r[None] + uv[:, 1:2] * self.u[None]

    def lift(self, uv, plane):
        D = self.rays(uv)
        n, d = plane[:3], plane[3]
        t = (d - n @ self.E) / (D @ n)
        return self.E + t[:, None] * D


def catmull_rom(P, n_per=80, alpha=0.5):
    """Центростремительный сплайн Катмулла–Рома через точки P (K,2)."""
    P = np.asarray(P, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(len(P) - 3):
        p0, p1, p2, p3 = P[i:i + 4]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        for t in np.linspace(t1, t2, n_per, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-2])
    return np.array(out)


def snap(g):
    return shapely.set_precision(g, GRID)


def silhouette(cam, stone):
    V, _ = stone.geometry()
    uv, _ = cam.project(V)
    return snap(MultiPoint(uv).convex_hull)


def front_facets(cam, stone):
    """Грани, смотрящие на глаз: (индекс плоскости, полигон в изображении)."""
    V, facets = stone.geometry()
    out = []
    for i, idx in facets:
        n = stone.P[i, :3]
        if n @ (cam.E - V[idx].mean(0)) <= 1e-9:
            continue
        uv, _ = cam.project(V[idx])
        poly = snap(Polygon(uv))
        if poly.area > 0:
            out.append((i, poly))
    return out


def depth_order(cam, stones, sil):
    """front[k] = список камней, которые заслоняют k (ближе к глазу)."""
    n = len(stones)
    front = [[] for _ in range(n)]
    tree = shapely.STRtree(sil)
    for k in range(n):
        for j in tree.query(sil[k]):
            j = int(j)
            if j <= k:
                continue
            inter = sil[k].intersection(sil[j])
            if inter.area < 1e-14:
                continue
            p = np.array(inter.representative_point().coords[0])
            D = cam.rays(p)[0]
            tk, tj = stones[k].ray_enter(cam.E, D), stones[j].ray_enter(cam.E, D)
            if tk is None or tj is None:
                continue
            if tj < tk:
                front[k].append(j)
            else:
                front[j].append(k)
    return front


def moss_regions(cam, stones, pattern, margin_mm=0.8):
    """Для каждого камня — область мха в плоскости изображения (с запасом под соседями)."""
    sil = [silhouette(cam, s) for s in stones]
    front = depth_order(cam, stones, sil)
    regions, visible = [], []
    for k, s in enumerate(stones):
        S = sil[k]
        if not S.intersects(pattern):
            regions.append(None)
            visible.append(None)
            continue
        occ = unary_union([sil[j] for j in front[k]]) if front[k] else None
        vis = S.difference(occ) if occ is not None else S
        vis = snap(vis)
        V, _ = s.geometry()
        _, dep = cam.project(V)
        m_img = margin_mm / float(dep.mean())
        grown = snap(vis.buffer(m_img, quad_segs=16).intersection(S))
        M = snap(pattern.intersection(grown))
        regions.append(M if M.area > 1e-12 else None)
        visible.append(vis)
    return regions, visible, sil, front


def _rings(poly):
    yield poly.exterior
    yield from poly.interiors


def _polys(g):
    return [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and p.area > 0]


def to_manifold(V, F):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(V, np.float32),
                               tri_verts=np.asarray(F, np.uint32)))


def from_manifold(m):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, np.int64)


def convex_manifold(stone):
    import manifold3d as mf
    V, _ = stone.geometry()
    return mf.Manifold.hull_points(np.asarray(V, float))


def moss_solid(cam, stone, M, facets, t_n=0.35, d_max=1.0, sink=0.12, min_area=0.02):
    """Площадка мха на камне: по призме на каждую грань, стенки — вдоль лучей к глазу.

    Призма = гомотетия грани с центром в глазу: верх параллелен грани на
    расстоянии t (по нормали), низ утоплен в камень на sink. Соседние призмы
    стыкуются по плоскостям «глаз + ребро», поэтому из точки обзора швов нет.
    """
    import manifold3d as mf
    parts = []
    for i, Fp in facets:
        Q = M.intersection(Fp)
        polys = _polys(Q)
        if not polys:
            continue
        plane = stone.P[i]
        n = plane[:3]
        h = float(n @ cam.E - plane[3])
        if h <= 1e-6:
            continue
        F0 = cam.E - h * n
        e = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
        a = unit(np.cross(n, e))
        b = np.cross(n, a)
        rings = []
        area3 = 0.0
        for p in polys:
            for r in _rings(p):
                uv = np.asarray(r.coords)[:-1]
                X = cam.lift(uv, plane)
                q = np.stack([(X - F0) @ a, (X - F0) @ b], 1)
                rings.append(q)
                ar = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
                area3 += ar if r.equals(p.exterior) else -ar
        if area3 < min_area:
            continue
        c = np.array(Q.centroid.coords[0])
        Xc = cam.lift(c[None], plane)[0]
        cos = float(n @ unit(cam.E - Xc))
        t = min(t_n, d_max * cos)
        s_top = 1 - t / h
        s_bot = 1 + sink / h
        cs = mf.CrossSection([r * s_bot for r in rings], mf.FillRule.EvenOdd).offset(0.02 * s_bot, mf.JoinType.Miter)
        pr = mf.Manifold.extrude(cs, (s_bot - s_top) * h, scale_top=(s_top / s_bot, s_top / s_bot))
        Fb = cam.E - s_bot * h * n
        T = np.array([[a[0], b[0], n[0], Fb[0]], [a[1], b[1], n[1], Fb[1]], [a[2], b[2], n[2], Fb[2]]])
        parts.append(pr.transform(T))
    if not parts:
        return None
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def moss_recess(cam, stone, M, facets, depth=1.0, lift=0.2, min_area=0.0):
    """Углубление под мох: по каждой грани — призма внутрь по нормали.

    Край углубления лежит на поверхности ровно по контуру узора (из точки
    обзора контур совпадает с рисунком), дно в тени — линия читается без краски.
    """
    import manifold3d as mf
    parts = []
    for i, Fp in facets:
        polys = _polys(M.intersection(Fp))
        if not polys:
            continue
        plane = stone.P[i]
        n = plane[:3]
        F0 = n * plane[3]
        e = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
        a = unit(np.cross(n, e))
        b = np.cross(n, a)
        rings = []
        for p in polys:
            for r in _rings(p):
                X = cam.lift(np.asarray(r.coords)[:-1], plane)
                rings.append(np.stack([(X - F0) @ a, (X - F0) @ b], 1))
        cs = mf.CrossSection(rings, mf.FillRule.EvenOdd).offset(0.02, mf.JoinType.Miter)
        if cs.area() <= min_area:
            continue
        pr = mf.Manifold.extrude(cs, depth + lift)
        o = F0 - depth * n
        T = np.array([[a[0], b[0], n[0], o[0]], [a[1], b[1], n[1], o[1]], [a[2], b[2], n[2], o[2]]])
        parts.append(pr.transform(T))
    if not parts:
        return None
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)
