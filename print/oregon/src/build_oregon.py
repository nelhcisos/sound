"""Орегон «ярость» — кот-воин (ортопоп), собран в коде (SDF) по картинкам Эмм_орегонярость/покой.png. Без генераций.
Высота ~100 мм (мастер), Z вверх, морда -> -Y.

usage: python3 build_oregon.py outdir [pitch=0.2]
Пишет: oregon_color.glb (цвета вершин), oregon_whole.stl, parts_truth.json.
"""
import sys, os, json
import numpy as np, trimesh
from skimage.measure import marching_cubes

OUT = sys.argv[1]; PITCH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
os.makedirs(OUT, exist_ok=True)
lo = np.array([-32.0, -27.0, -0.5]) + 0.137 * PITCH; hi = np.array([32.0, 30.0, 104.0])
xs, ys, zs = [np.arange(lo[i], hi[i] + PITCH, PITCH, dtype=np.float32) for i in range(3)]
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
print("сетка", X.shape, "точек %.1f млн" % (X.size / 1e6), flush=True)


def L(*a): return np.sqrt(sum(t * t for t in a))
def sphere(c, r): return L(X - c[0], Y - c[1], Z - c[2]) - r
def ellipsoid(c, r):
    px, py, pz = (X - c[0]) / r[0], (Y - c[1]) / r[1], (Z - c[2]) / r[2]
    k0 = L(px, py, pz); k1 = L(px / r[0], py / r[1], pz / r[2])
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)
def cone(a, b, ra, rb):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); ba = b - a; l2 = float(ba @ ba)
    px, py, pz = X - a[0], Y - a[1], Z - a[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / l2, 0, 1)
    return L(px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h) - (ra + (rb - ra) * h)
def cylz(c, r, z0, z1, rr=0.0):
    d1 = L(X - c[0], Y - c[1]) - (r - rr); d2 = np.abs(Z - (z0 + z1) / 2) - ((z1 - z0) / 2 - rr)
    return L(np.maximum(d1, 0), np.maximum(d2, 0)) + np.minimum(np.maximum(d1, d2), 0) - rr
def plane(p, n):
    n = np.asarray(n, np.float32); n = n / np.linalg.norm(n)
    return (X - p[0]) * n[0] + (Y - p[1]) * n[1] + (Z - p[2]) * n[2]
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1); return b * (1 - h) + a * h - k * h * (1 - h)
def smax(a, b, k): return -smin(-a, -b, k)
def U(*ds):
    out = ds[0]
    for d in ds[1:]: out = np.minimum(out, d)
    return out
def polyline(pts, r0, r1):
    d = None; n = len(pts) - 1
    for i in range(n):
        ra = r0 + (r1 - r0) * i / n; rb = r0 + (r1 - r0) * (i + 1) / n
        s = cone(pts[i], pts[i + 1], ra, rb); d = s if d is None else np.minimum(d, s)
    return d


class Acc(dict):
    def __init__(self):
        super().__init__(); self.best = None; self.owner = None; self.names = []
    def __setitem__(self, k, d):
        d = d.astype(np.float32, copy=False)
        if self.best is None:
            self.best = d.copy(); self.owner = np.zeros(d.shape, np.uint8)
        else:
            m_ = d < self.best; self.owner[m_] = len(self.names); np.minimum(self.best, d, out=self.best)
        self.names.append(k)
P = Acc()
COL = {"base": (236, 236, 242), "boot": (110, 112, 122), "gi": (150, 190, 236), "belt": (232, 120, 200),
       "mitten_L": (150, 212, 120), "mitten_R": (246, 176, 80), "head": (246, 226, 140), "muzzle": (252, 250, 240),
       "nose": (232, 120, 140), "tail": (170, 100, 220), "dark": (60, 50, 50)}

# подставка
P["base"] = cylz((0, 0), 24.0, 0.0, 5.0, 1.4)
# сапоги: большие круглые, носок вперёд
bt = []
for s in (-1, 1):
    b = smin(ellipsoid((8.0 * s, -2.0, 12.6), (8.6, 10.0, 8.2)), cylz((8.0 * s, 0.5), 6.6, 14.0, 22.5, 2.0), 3.0)
    b = np.maximum(b, -(Z - 4.6))
    b = np.minimum(b, np.maximum(ellipsoid((8.0 * s, 0.5, 21.6), (7.6, 7.6, 1.9)), -(Z - 20.0)))     # отворот-валик сверху
    bt.append(b)
P["boot_R"], P["boot_L"] = bt[0], bt[1]
# кимоно: единый силуэт по профилю (юбка шире книзу, талия у пояса, грудь, плечи); низ и верх перекрываются под поясом
zk = np.array([20.5, 26.0, 32.0, 38.4, 44.0, 50.0, 55.0, 58.5]); rxk = np.array([16.0, 15.2, 13.6, 11.6, 12.8, 13.2, 11.8, 8.0])
ryk = rxk * 0.80
rx = np.interp(Z, zk, rxk).astype(np.float32); ry = np.interp(Z, zk, ryk).astype(np.float32)
body = (L(X / rx, (Y - 0.5) / ry) - 1.0) * np.minimum(rx, ry)
body = smax(body, -(Z - 20.5), 1.0); body = smax(body, Z - 58.5, 2.0)
P["gi_bottom"] = np.maximum(np.maximum(body, Z - 39.4), -np.maximum(np.abs(X - 2.0 + 0.12 * (Z - 20)) - 0.25, Y + 10.0))   # бороздка запаха спереди
# пояс: кольцо + узел с концами спереди
belt = np.maximum((L(X / 13.6, (Y - 0.5) / 11.1) - 1.0) * 11.1, np.abs(Z - 38.4) - 1.8)          # кольцо-пояс
belt = smax(belt, -((L(X / 11.2, (Y - 0.5) / 8.9) - 1.0) * 8.9), 0.3)
knot = U(ellipsoid((-1.0, -11.2, 38.4), (2.6, 1.6, 2.2)),
         polyline([(-0.5, -11.8, 37.2), (-1.6, -12.6, 33.0), (-3.0, -13.4, 29.0)], 1.3, 1.0),
         polyline([(0.3, -11.8, 37.2), (1.8, -12.7, 33.6), (3.4, -13.4, 30.4)], 1.3, 1.0))
P["belt"] = U(belt, knot)
# кимоно-верх: грудь, отворот запаха, широкие короткие рукава
gt = np.maximum(body, -(Z - 37.4))
lapel = np.maximum(np.maximum(np.abs(plane((0, -9.0, 56), (0.78, 0, 0.62))) - 1.3, body - 0.9), Y + 2.0)   # отворот прижат к груди
gt = np.minimum(gt, np.maximum(lapel, np.maximum(-(Z - 41.5), Z - 57.0)))
for s in (-1, 1):
    sl = cone((10.5 * s, 0.5, 52.0), (17.5 * s, -0.5, 44.0), 5.2, 6.4)
    sl = smax(sl, plane((17.5 * s, -0.5, 44.0), (0.66 * s, 0, -0.75)), 0.5)
    gt = smin(gt, sl, 1.6)
P["gi_top"] = gt
# рукавицы-лапы: огромные, рукав входит в манжету
for nm, s in (("mitten_R", -1), ("mitten_L", 1)):
    c = np.array([22.5 * s, -2.0, 36.0])
    m = ellipsoid(c, (7.4, 7.0, 8.2))
    m = smin(m, cylz((19.4 * s, -0.6), 5.4, 40.0, 45.8, 1.6), 1.2)                                  # манжета
    for k, (dx, dz) in enumerate([(-3.2, -6.5), (0.0, -7.4), (3.2, -6.5)]):                        # пальцы-подушечки
        m = smin(m, sphere((c[0] + dx * 0.6 * s, c[1] - 5.4, c[2] + dz * 0.55), 2.6), 1.0)
    m = smin(m, ellipsoid((c[0] - 6.0 * s, c[1] - 3.0, c[2] + 1.0), (2.6, 2.4, 3.6)), 1.2)          # большой палец
    P[nm] = m
# голова: большая кошачья, уши, меховые щёки, сердитые глаза
hd = ellipsoid((0, 0.5, 76.0), (21.0, 17.0, 18.0))
hd = smin(hd, cylz((0, 0.5), 6.0, 55.0, 62.0, 1.5), 2.5)
for s in (-1, 1):
    ear = smin(cone((12.0 * s, 1.5, 86.0), (17.0 * s, 2.5, 99.0), 7.6, 1.8), ellipsoid((13.0 * s, 1.8, 89.0), (6.0, 4.4, 6.0)), 2.0)
    ear = np.maximum(ear, -np.maximum(cone((12.6 * s, 0.0, 88.0), (16.6 * s, 1.0, 97.5), 4.8, 0.8), -(-(Y - 0.2))))   # ямка уха
    hd = smin(hd, ear, 2.2)
    for k in range(3):                                                                              # пушистые щёки
        t = k / 2.0
        tuft = cone((17.5 * s, -5.0 + 3 * t, 71.0 - 4 * t), (23.0 * s, -4.0 + 3 * t, 67.5 - 5 * t), 3.0, 0.8)
        hd = smin(hd, tuft, 1.6)
    eye = polyline([(4.0 * s, -16.4, 77.5), (7.5 * s, -16.0, 79.4), (11.0 * s, -14.6, 80.6)], 0.55, 0.45)
    hd = np.maximum(hd, -eye)                                                                       # прищур
    hd = np.minimum(hd, polyline([(3.6 * s, -16.0, 81.8), (8.0 * s, -15.2, 84.2), (12.0 * s, -13.4, 83.6)], 0.8, 0.55))   # сердитые брови
for i, x in enumerate((-3.0, 0.0, 3.0)):                                                            # хохолок на макушке
    hd = smin(hd, cone((x, -2.0, 92.0), (x * 1.5, -4.0 - abs(x) * 0.3, 98.5 - abs(x) * 0.6), 2.6, 0.8), 1.0)
P["head"] = hd
# мордочка: белая, выступает; нос; рот
mz = ellipsoid((0, -15.6, 70.0), (9.0, 4.4, 6.0))
for s in (-1, 1):
    mz = smin(mz, ellipsoid((3.6 * s, -16.4, 69.4), (4.4, 3.4, 3.8)), 1.0)
    for k in range(3):                                                                              # усы — бороздки
        mz = np.maximum(mz, -polyline([(4.0 * s, -19.6, 70.5 - 1.2 * k), (8.0 * s, -18.6, 71.0 - 1.6 * k)], 0.3, 0.3))
mz = np.maximum(mz, -polyline([(-3.0, -19.6, 66.8), (0.0, -20.2, 67.6), (3.0, -19.6, 66.8)], 0.4, 0.4))   # рот
P["muzzle"] = mz
P["nose"] = ellipsoid((0, -19.9, 71.6), (1.9, 1.3, 1.3))
# хвост: из-под пояса сзади, изгиб вбок и вверх, кончик-сердечко
tail_pts = [(2.0, 10.0, 32.0), (8.0, 15.0, 30.0), (16.0, 17.5, 32.0), (23.0, 16.5, 38.0), (26.5, 13.5, 45.0)]
tail = polyline(tail_pts, 2.6, 2.0)
tip = U(sphere((26.0, 12.6, 48.6), 2.7), sphere((29.0, 12.6, 47.8), 2.7), cone((27.6, 12.8, 47.0), (27.6, 12.8, 43.6), 2.6, 0.6))
P["tail"] = smin(tail, tip, 1.0)

names = P.names; F = P.best; OWN = P.owner
print("поверхность…", flush=True)
v, f, _, _ = marching_cubes(F, 0.0, spacing=(PITCH, PITCH, PITCH))
v = v + lo
m = trimesh.Trimesh(v, f[:, ::-1], process=True)
if m.volume < 0: m.invert()
m = max(m.split(only_watertight=False), key=lambda c: c.area)
idx = np.clip(np.round((m.vertices - lo) / PITCH).astype(int), 0, np.array(F.shape) - 1)
owner = OWN[idx[:, 0], idx[:, 1], idx[:, 2]]
key = lambda n: {"boot_L": "boot", "boot_R": "boot", "gi_bottom": "gi", "gi_top": "gi"}.get(n, n)
cols = np.array([COL[key(n)] for n in names], np.uint8)
m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.c_[cols[owner], np.full(len(owner), 255, np.uint8)])
print("граней %d, замкнута %s, размер %s" % (len(m.faces), m.is_watertight, m.extents.round(1)), flush=True)
m.export(os.path.join(OUT, "oregon_color.glb"))
trimesh.Trimesh(m.vertices, m.faces).export(os.path.join(OUT, "oregon_whole.stl"))
json.dump({"parts": names, "colors": {n: COL[key(n)] for n in names}}, open(os.path.join(OUT, "parts_truth.json"), "w"), ensure_ascii=False, indent=1)
print("ok", OUT)
