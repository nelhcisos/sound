"""Эмм «в эластиках» (v2) — детальная гладкая фигурка в коде (SDF), по картинке Эмм_вэластиках.png и памятке CLAUDE.md.
Высота ~100 мм (мастер), Z вверх, лицо -> -Y. Без генераций.

usage: python3 build_emm_v2.py outdir [pitch=0.2]
Пишет: emm_color.glb (цвета вершин), emm_whole.stl, parts_truth.json.
Одежда соединяется с телом жёстко (min) — у краёв настоящие ступеньки, как у хорошей модели.
"""
import sys, os, json
import numpy as np, trimesh
from skimage.measure import marching_cubes

OUT = sys.argv[1]; PITCH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
os.makedirs(OUT, exist_ok=True)
lo = np.array([-27.0, -26.0, -0.5]) + 0.137 * PITCH; hi = np.array([27.0, 29.5, 104.0])
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
def rbox(c, half, r):
    qx = np.abs(X - c[0]) - half[0] + r; qy = np.abs(Y - c[1]) - half[1] + r; qz = np.abs(Z - c[2]) - half[2] + r
    return L(np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r
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
def gauss(z, z0, w): return np.exp(-((Z - z0) / w) ** 2) if z is None else np.exp(-((z - z0) / w) ** 2)
def polyline(pts, r0, r1):
    """гладкая «нитка» по точкам, радиус от r0 к r1"""
    d = None; n = len(pts) - 1
    for i in range(n):
        ra = r0 + (r1 - r0) * i / n; rb = r0 + (r1 - r0) * (i + 1) / n
        s = cone(pts[i], pts[i + 1], ra, rb); d = s if d is None else np.minimum(d, s)
    return d


class Acc(dict):
    """деталь сразу вливается в общее поле (min) и карту владельцев — поля не копятся в памяти"""
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
COL = {"base": (232, 226, 242), "croc": (176, 214, 240), "pants": (176, 212, 236), "top": (186, 220, 242),
       "skin": (248, 216, 194), "glove": (170, 206, 238), "mask": (252, 252, 252), "hair": (250, 250, 252),
       "cap": (184, 218, 240), "lash": (30, 30, 34), "loop": (252, 252, 252)}

# ---------- подставка ----------
P["base"] = cylz((0, 0), 23.0, 0.0, 5.0, 1.4)

# ---------- кроксы: башмак, дырочки, ремешок с заклёпками ----------
cr = []
for s in (-1, 1):
    x = 6.6 * s
    b = smin(ellipsoid((x, -2.6, 8.6), (5.3, 8.8, 4.3)), ellipsoid((x, 3.2, 8.1), (4.7, 4.9, 3.5)), 1.5)
    b = np.maximum(b, -(Z - 4.6))
    b = np.maximum(b, -sphere((x, 1.4, 14.8), 4.7))
    for hx, hy in [(-2.5, -8.2), (0, -9.4), (2.5, -8.2), (-3.1, -5.3), (0, -6.3), (3.1, -5.3), (-3.3, -2.4), (3.3, -2.4), (-1.6, -3.9), (1.6, -3.9)]:
        b = np.maximum(b, -sphere((x + hx, -2.6 + hy * 0.75, 12.7 - abs(hy) * 0.12), 0.78))
    th = np.linspace(-1.25, 1.25, 7)
    strap = polyline([(x + 5.0 * np.sin(t), 2.2 + 4.2 * np.cos(t), 9.6 + 0.5 * np.cos(t)) for t in th], 0.85, 0.85)
    rivets = U(sphere((x - 5.0 * np.sin(1.25), 2.2 + 4.2 * np.cos(1.25), 9.9), 1.0), sphere((x + 5.0 * np.sin(1.25), 2.2 + 4.2 * np.cos(1.25), 9.9), 1.0))
    cr.append(U(b, strap, rivets))
P["croc_R"], P["croc_L"] = cr[0], cr[1]

# ---------- штаны-карго: сборка на резинке у щиколоток, карманы с клапанами ----------
legs = None
for s in (-1, 1):
    xc = 5.9 * s
    lg = cone((xc, 0.7, 12.0), (5.6 * s, 0.2, 32.0), 5.0, 5.9)
    th = np.arctan2(Y - 0.6, X - xc)
    puff = 0.95 * gauss(None, 15.6, 1.6) - 0.45 * gauss(None, 13.2, 0.55)          # пышно над резинкой, утяжка на ней
    ridges = 0.32 * np.cos(16 * th) * np.clip(1 - np.abs(Z - 15.3) / 2.4, 0, 1)      # складочки сборки
    lg = lg - puff - ridges
    legs = lg if legs is None else smin(legs, lg, 2.0)
pants = smin(legs, ellipsoid((0, 0.2, 34.0), (11.4, 7.8, 6.5)), 3.0)
for s in (-1, 1):
    pants = np.minimum(pants, rbox((10.9 * s, 0.4, 22.0), (1.0, 3.4, 3.4), 0.7))        # карман
    pants = np.minimum(pants, rbox((11.25 * s, 0.4, 25.6), (1.0, 3.7, 0.9), 0.45))      # клапан
    pants = np.minimum(pants, sphere((11.95 * s, 0.4, 25.4), 0.55))                      # кнопка
pants = np.minimum(pants, U(cone((-1.8, -6.7, 33.0), (-2.6, -5.8, 26.8), 1.0, 0.9), cone((1.8, -6.7, 33.0), (2.6, -5.8, 26.8), 1.0, 0.9)))
P["pants"] = pants

# ---------- хирургичка: V-вырез, кармашек, пышные короткие рукава, сборка на резинке на талии ----------
tz = (Z - 31.0) / 27.0
half_x = 14.4 - 2.2 * np.clip(tz, 0, 1); half_y = 9.6 - 1.2 * np.clip(tz, 0, 1)
top = L(np.maximum(np.abs(X) / half_x * 10 - 10 + 2.5, 0), np.maximum(np.abs(Y - 0.3) / half_y * 7 - 7 + 2.5, 0)) - 2.5
top = np.maximum(top, np.abs(Z - 45.0) - 14.0)
th = np.arctan2(Y - 0.3, X)
gather = 0.7 * gauss(None, 41.8, 1.5) + 0.55 * gauss(None, 37.6, 1.5) - 0.95 * gauss(None, 39.7, 0.6)   # пыш-утяжка-пыш
top = top - gather - 0.34 * np.cos(30 * th) * np.clip(1 - np.abs(Z - 39.7) / 3.6, 0, 1)
vneck = np.maximum(np.maximum(-plane((0, -10, 50.5), (0.82, 0, 0.57)), -plane((0, -10, 50.5), (-0.82, 0, 0.57))), Y + 3.5)
top = np.maximum(top, -vneck)
top = np.maximum(top, -cylz((0, 0.3), 5.0, 54.0, 62.0))
top = np.minimum(top, rbox((6.2, -9.3, 47.2), (3.0, 0.7, 3.0), 0.5))
for s in (-1, 1):
    sl = cone((11.0 * s, 0.2, 54.0), (16.3 * s, 0.0, 44.5), 5.6, 5.1)
    sl = smax(sl, plane((16.3 * s, 0.0, 44.5), (0.48 * s, 0, -0.88)), 0.4)
    top = smin(top, sl, 1.5)
P["top"] = top

# ---------- руки и перчатки (длинная манжета) ----------
arms = []; gloves = []
for s in (-1, 1):
    arms.append(cone((15.2 * s, 0.0, 48.0), (19.8 * s, -0.8, 33.8), 2.9, 2.6))
    g = smin(ellipsoid((20.4 * s, -1.0, 30.2), (3.0, 2.3, 3.8)), cone((19.6 * s, -0.7, 36.4), (20.2 * s, -0.9, 31.5), 3.15, 2.9), 1.0)
    g = smin(g, cone((18.6 * s, -2.6, 31.0), (18.4 * s, -3.4, 28.6), 0.95, 0.85), 0.6)
    gloves.append(g)
P["arm_R"], P["arm_L"] = arms[0], arms[1]
P["glove_R"], P["glove_L"] = gloves[0], gloves[1]

# ---------- голова: шея, уши, закрытые глаза с ресницами, петли маски ----------
neck = cylz((0, 0.6), 4.4, 54.0, 63.0, 0.8)
head = ellipsoid((0, 0.5, 76.5), (17.6, 15.8, 16.2))
head = smin(head, neck, 2.0)
for s in (-1, 1):
    head = smin(head, ellipsoid((17.0 * s, 2.0, 75.0), (2.3, 3.1, 3.8)), 0.8)
lash = None
for s in (-1, 1):
    cx = 6.6 * s
    arc = [(cx + t, -14.75 + 0.02 * t * t, 77.6 - 0.16 * t * t) for t in np.linspace(-3.2, 3.2, 9)]
    head = np.maximum(head, -polyline(arc, 0.42, 0.42))                                     # линия закрытого глаза
    lid = polyline([(cx + t, -14.95 + 0.02 * t * t, 77.45 - 0.16 * t * t) for t in np.linspace(-3.4, 3.4, 9)], 0.55, 0.55)
    for t in np.linspace(-2.8, 3.2, 6):                                                      # ресницы вниз-наружу
        p0 = (cx + t, -14.95 + 0.02 * t * t, 77.3 - 0.16 * t * t)
        p1 = (cx + t * 1.12 + 0.55 * s, p0[1] - 0.9, p0[2] - 1.5)
        lid = np.minimum(lid, cone(p0, p1, 0.42, 0.16))
    lash = lid if lash is None else np.minimum(lash, lid)
    head = np.maximum(head, -cone((s * 4.0, -14.9, 81.2), (s * 8.4, -14.0, 81.7), 0.3, 0.24))   # брови
    for zz in (71.6, 66.2):                                                                   # петли маски к уху
        lp = polyline([(11.4 * s, -11.0, zz), (14.6 * s, -6.5, zz + (75.0 - zz) * 0.45), (16.4 * s, -1.0, 75.0 - (75.0 - zz) * 0.25), (16.9 * s, 2.0, 75.2)], 0.5, 0.44)
        loops = lp if s == -1 and zz == 71.6 else np.minimum(loops, lp)
P["face"] = head
P["loop"] = loops
P["lash"] = lash

# ---------- маска: толстая, со складками ----------
outer = ellipsoid((0, 0.5, 76.5), (19.4, 17.6, 18.0))
mask = smax(outer, -(-Y - 4.0), 0.8); mask = smax(mask, np.abs(Z - 68.8) - 5.0, 1.2)
mask = smax(mask, np.abs(X) - 11.8, 1.6); mask = smax(mask, -ellipsoid((0, 0.5, 76.5), (17.0, 15.2, 15.6)), 0.3)
for zz in (66.6, 68.8, 71.0):
    mask = smax(mask, -np.maximum(np.abs(Z - zz) - 0.3, -(outer + 0.4)), 0.15)
P["mask"] = mask

# ---------- шапка-берет: мешковатая, сползает назад, толстый подвёрнутый валик ----------
edge = 82.6 - 0.45 * (Y + 16.0) + 0.016 * X * X                                              # край шапки
capE = ellipsoid((0, 4.6, 82.0), (20.2, 21.2, 19.6))
thc = np.arctan2(Y - 4.6, X); phc = np.clip((Z - 84.0) / 14.0, 0, 1)
capE = capE - 0.55 * np.sin(5 * thc + 1.3 * phc) * phc * (1 - phc) * 4 * 0.5            # мягкие складки купола
cap = np.maximum(capE, -(Z - edge))
cap = smax(cap, -ellipsoid((0, 0.5, 76.5), (17.2, 15.4, 15.8)), 0.4)
roll = smax(ellipsoid((0, 4.6, 82.0), (21.6, 22.6, 21.0)), np.abs(Z - (edge + 1.9)) - 1.9, 0.9)
roll = smax(roll, -ellipsoid((0, 0.5, 76.5), (17.2, 15.4, 15.8)), 0.6)
cap = smin(cap, roll, 0.8)
cap = np.maximum(cap, -np.maximum(np.abs(Z - (edge + 3.8)) - 0.22, -(capE + 0.3)))           # бороздка над валиком
knot = U(ellipsoid((-3.0, 24.6, 79.0), (3.0, 1.8, 1.9)), ellipsoid((3.0, 24.6, 79.0), (3.0, 1.8, 1.9)), sphere((0, 24.5, 79.0), 1.6))
P["cap"] = np.minimum(cap, knot)

# ---------- белые колючие волосы из-под шапки: чёлка, у висков, на затылке ----------
hair = None
def spike(root, tip, r0=1.6, r1=0.62, bend=(0, -0.6, 0.0)):
    mid = tuple(0.5 * (root[i] + tip[i]) + bend[i] for i in range(3))   # прядь с изгибом наружу
    return polyline([root, mid, tip], r0, r1)
rng = np.random.default_rng(7)
for i, x in enumerate(np.linspace(-13.5, 13.5, 17)):                                     # чёлка
    y0 = -14.5 + 0.042 * x * x; zt = 84.4 - 0.02 * x * x
    dz = rng.uniform(0, 2.2); dx = rng.uniform(-1.2, 1.2)
    tip = (x * 1.06 + dx, y0 - 2.4, max(79.2 - dz - 0.035 * x * x, 75.2))      # не ниже 75: ресницы и петли маски свободны
    sp = spike((x, y0 + 1.6, zt), tip, 1.6 + rng.uniform(-0.2, 0.2), 0.6, (0, -0.9, 0.3))
    hair = sp if hair is None else smin(hair, sp, 0.6)
for s in (-1, 1):                                                                           # у висков, перед ушами
    for k, (yy, zz) in enumerate([(-6.5, 80.5), (-3.0, 81.0)]):
        root = (15.6 * s, yy, zz); tip = (17.6 * s, yy - 1.5, 75.4 - 0.4 * k)
        hair = smin(hair, spike(root, tip, 1.7, 0.62, (0.5 * s, -0.4, 0)), 0.6)
for i, x in enumerate(np.linspace(-11.0, 11.0, 11)):                                      # затылок из-под шапки
    root = (x, 13.0 - 0.03 * x * x, 70.8); tip = (x * 1.1, 15.6 - 0.03 * x * x, 64.6 - rng.uniform(0, 1.6))
    hair = smin(hair, spike(root, tip, 1.8, 0.62, (0, 0.7, 0)), 0.6)
P["hair"] = hair

# ---------- сборка ----------
names = P.names; F = P.best; OWN = P.owner
print("поверхность…", flush=True)
v, f, _, _ = marching_cubes(F, 0.0, spacing=(PITCH, PITCH, PITCH))
v = v + lo
m = trimesh.Trimesh(v, f[:, ::-1], process=True)
if m.volume < 0: m.invert()
m = max(m.split(only_watertight=False), key=lambda c: c.area)
idx = np.clip(np.round((m.vertices - lo) / PITCH).astype(int), 0, np.array(F.shape) - 1)
owner = OWN[idx[:, 0], idx[:, 1], idx[:, 2]]
key = lambda n: {"croc_L": "croc", "croc_R": "croc", "arm_L": "skin", "arm_R": "skin", "glove_L": "glove",
                 "glove_R": "glove", "face": "skin"}.get(n, n)
cols = np.array([COL[key(n)] for n in names], np.uint8)
m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.c_[cols[owner], np.full(len(owner), 255, np.uint8)])
print("граней %d, замкнута %s, размер %s" % (len(m.faces), m.is_watertight, m.extents.round(1)), flush=True)
m.export(os.path.join(OUT, "emm_color.glb"))
trimesh.Trimesh(m.vertices, m.faces).export(os.path.join(OUT, "emm_whole.stl"))
json.dump({"parts": names, "colors": {n: COL[key(n)] for n in names},
           "faces_per_part": {n: int((owner[m.faces[:, 0]] == i).sum()) for i, n in enumerate(names)}},
          open(os.path.join(OUT, "parts_truth.json"), "w"), ensure_ascii=False, indent=1)
print("ok", OUT)
