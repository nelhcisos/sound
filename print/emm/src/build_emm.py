"""Эмм (ортопоп) — фигурка, собранная в коде из гладких форм (SDF), без генераций.
Памятка внешности — CLAUDE.md, раздел «Персонаж Эмм». Высота 100 мм (мастер), Z вверх, лицо -> -Y.

usage: python3 build_emm.py outdir [pitch=0.25]
Пишет: outdir/emm_color.glb (цвета вершин, как «модель с текстурой»), outdir/emm_whole.stl, outdir/parts_truth.json.
Одежда соединяется с телом жёстко (min), чтобы у краёв были настоящие ступеньки — как у модели из генератора.
"""
import sys, os, json
import numpy as np, trimesh
from skimage.measure import marching_cubes

OUT = sys.argv[1]; PITCH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
os.makedirs(OUT, exist_ok=True)
lo = np.array([-27.0, -26.0, -0.5]) + 0.137 * PITCH; hi = np.array([27.0, 26.0, 100.5])  # сдвиг: поверхности не на узлах сетки
xs, ys, zs = [np.arange(lo[i], hi[i] + PITCH, PITCH, dtype=np.float32) for i in range(3)]
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
print("сетка", X.shape, "точек %.1f млн" % (X.size / 1e6))


# ---------- примитивы ----------
def L(*a): return np.sqrt(sum(t * t for t in a))
def sphere(c, r): return L(X - c[0], Y - c[1], Z - c[2]) - r
def ellipsoid(c, r):
    px, py, pz = (X - c[0]) / r[0], (Y - c[1]) / r[1], (Z - c[2]) / r[2]
    k0 = L(px, py, pz); k1 = L(px / r[0], py / r[1], pz / r[2])
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)
def cone(a, b, ra, rb):
    """капсула с разным радиусом на концах"""
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


# ---------- детали (каждая — своё поле и свой цвет) ----------
P = {}
COL = {"base": (226, 214, 240), "croc": (120, 185, 235), "pants": (130, 180, 225), "top": (165, 210, 240),
       "skin": (247, 214, 190), "glove": (95, 160, 235), "mask": (250, 250, 250), "bangs": (252, 252, 255),
       "cap": (150, 205, 240)}
# подставка
P["base"] = cylz((0, 0), 24.0, 0.0, 5.0, 1.2)
# кроксы: пухлый башмак, открытая пятка, дырочки сверху
cr = []
for s in (-1, 1):
    x = 6.4 * s
    b = smin(ellipsoid((x, -2.5, 8.6), (5.2, 8.6, 4.2)), ellipsoid((x, 3.0, 8.0), (4.6, 4.8, 3.4)), 1.5)
    b = np.maximum(b, -(Z - 4.6))                       # плоская подошва, чуть утоплена в подставку
    b = np.maximum(b, -sphere((x, 1.2, 14.6), 4.6))      # вход для ноги
    for i, (hx, hy) in enumerate([(-2.4, -8), (0, -9.2), (2.4, -8), (-3, -5.2), (0, -6.2), (3, -5.2), (-3.2, -2.4), (3.2, -2.4)]):
        b = np.maximum(b, -sphere((x + hx, -2.5 + hy * 0.75, 12.6 - abs(hy) * 0.12), 0.75))
    cr.append(b)
P["croc_L"], P["croc_R"] = cr[1], cr[0]   # «L» — левая нога персонажа (у него слева = у нас справа, x>0)
# штаны: две штанины + таз, карманы-карго, завязки
legs = smin(cone((6.0, 0.6, 11.5), (5.6, 0.2, 31.0), 4.6, 5.6), cone((-6.0, 0.6, 11.5), (-5.6, 0.2, 31.0), 4.6, 5.6), 2.0)
pants = smin(legs, ellipsoid((0, 0.2, 33.0), (11.2, 7.6, 6.5)), 3.0)
for s in (-1, 1):
    pants = np.minimum(pants, rbox((10.5 * s, 0.2, 22.5), (1.1, 3.4, 3.2), 0.7))   # карманы на бёдрах
# завязки на поясе: лежат на штанах, свисают из-под подола
pants = np.minimum(pants, U(cone((-1.8, -6.6, 33.0), (-2.6, -5.6, 26.5), 1.0, 0.9), cone((1.8, -6.6, 33.0), (2.6, -5.6, 26.5), 1.0, 0.9)))
P["pants"] = pants
# хирургичка оверсайз: трапеция, подол нависает над штанами, V-вырез, кармашек, короткие широкие рукава
tz = (Z - 31.0) / 27.0
half_x = 14.6 - 2.4 * np.clip(tz, 0, 1); half_y = 9.6 - 1.2 * np.clip(tz, 0, 1)
top = L(np.maximum(np.abs(X) / half_x * 10 - 10 + 2.5, 0), np.maximum(np.abs(Y - 0.3) / half_y * 7 - 7 + 2.5, 0)) - 2.5
top = np.maximum(top, np.abs(Z - 45.0) - 14.0)
top = smax(top, -(Z - 31.0), 0.6)
vneck = np.maximum(np.maximum(-plane((0, -10, 50.5), (0.82, 0, 0.57)), -plane((0, -10, 50.5), (-0.82, 0, 0.57))), Y + 3.5)
top = np.maximum(top, -vneck)
top = np.maximum(top, -cylz((0, 0.3), 5.0, 54.0, 62.0))                 # горловина
top = np.minimum(top, rbox((6.2, -9.3, 46.5), (3.0, 0.7, 3.2), 0.5))     # кармашек на левой груди персонажа (он смотрит на нас: его лево = +x)
for s in (-1, 1):
    sl = cone((11.0 * s, 0.2, 54.0), (16.3 * s, 0.0, 44.5), 5.4, 4.9)
    sl = smax(sl, plane((16.3 * s, 0.0, 44.5), (0.48 * s, 0, -0.88)), 0.4)  # срез рукава
    top = smin(top, sl, 1.5)
P["top"] = top
# руки (кожа) из рукавов + перчатки
arms = []; gloves = []
for s in (-1, 1):
    arms.append(cone((15.2 * s, 0.0, 48.0), (19.8 * s, -0.8, 33.8), 2.9, 2.6))   # чуть в стороны: зазор 1,5 мм до тела
    g = smin(ellipsoid((20.4 * s, -1.0, 30.2), (3.0, 2.3, 3.8)), cone((19.9 * s, -0.8, 34.6), (20.2 * s, -0.9, 31.5), 3.05, 2.9), 1.0)
    g = smin(g, cone((18.6 * s, -2.6, 31.0), (18.4 * s, -3.4, 28.6), 0.95, 0.85), 0.6)   # большой палец
    gloves.append(g)
P["arm_R"], P["arm_L"] = arms[0], arms[1]          # x<0 — правая рука персонажа
P["glove_R"], P["glove_L"] = gloves[0], gloves[1]
# шея и голова
neck = cylz((0, 0.6), 4.4, 54.0, 63.0, 0.8)
head = ellipsoid((0, 0.5, 76.5), (17.6, 15.8, 16.2))
head = smin(head, neck, 2.0)
for s in (-1, 1):
    head = smin(head, ellipsoid((17.0 * s, 2.0, 75.0), (2.3, 3.1, 3.8)), 0.8)        # уши
# закрытые глаза — тонкие дуги-гравировка, брови — лёгкий рельеф
for s in (-1, 1):
    for t0, t1 in ((-3.0, -1.0), (-1.0, 1.0), (1.0, 3.0)):
        a = (s * 6.5 + t0, -14.6, 77.3 - 0.12 * t0 * t0); b = (s * 6.5 + t1, -14.6, 77.3 - 0.12 * t1 * t1)
        head = np.maximum(head, -cone(a, b, 0.45, 0.45))
    head = np.maximum(head, -cone((s * 4.0, -14.9, 80.4), (s * 8.4, -14.0, 80.9), 0.32, 0.25))   # брови — тонкая гравировка
P["face"] = head
# маска: толстый белый «намордник» на нос и рот, со складками
shell = ellipsoid((0, 0.5, 76.5), (19.4, 17.6, 18.0))
mask = np.maximum(shell, -(-Y - 4.0)); mask = np.maximum(mask, np.abs(Z - 68.6) - 5.6)
mask = np.maximum(mask, np.abs(X) - 12.0); mask = smax(mask, -ellipsoid((0, 0.5, 76.5), (17.0, 15.2, 15.6)), 0.3)
outer = ellipsoid((0, 0.5, 76.5), (19.4, 17.6, 18.0))
for zz in (66.4, 68.8, 71.2):                        # складки — бороздки по поверхности маски
    mask = np.maximum(mask, -np.maximum(np.abs(Z - zz) - 0.35, -(outer + 0.45)))
P["mask"] = mask
# шапочка: мешковатая, край спереди над чёлкой, сзади ниже; узел сзади
cap = ellipsoid((0, 1.4, 79.5), (19.2, 17.4, 19.0))
cap = np.maximum(cap, -(Z - (84.0 - 0.5 * (Y + 16.0) + 0.02 * X * X)))   # край: спереди над чёлкой, по бокам над ушами, сзади ниже
cap = smax(cap, -ellipsoid((0, 0.5, 76.5), (17.2, 15.4, 15.8)), 0.4)
knot = U(ellipsoid((-3.2, 18.4, 75.5), (3.0, 1.8, 1.9)), ellipsoid((3.2, 18.4, 75.5), (3.0, 1.8, 1.9)), sphere((0, 18.3, 75.5), 1.6),
         cone((-0.8, 18.0, 74.5), (-2.2, 18.6, 69.5), 0.9, 0.7), cone((0.8, 18.0, 74.5), (2.4, 18.6, 69.8), 0.9, 0.7))
cap = np.minimum(cap, knot)
P["cap"] = cap
# чёлка: белые колючие пряди из-под шапочки на лоб
bangs = None
for i, x in enumerate(np.linspace(-11.5, 11.5, 9)):
    y0 = -14.8 + 0.035 * x * x; zt = 85.0 - 0.03 * x * x
    tip = (x * 1.08 + (0.8 if i % 2 else -0.8), y0 - 2.4, 79.8 - (1.0 if i % 2 else 0.0) - 0.03 * x * x)
    sp = cone((x, y0 + 1.4, zt), tip, 2.2, 0.85)        # кончик 0,85 мм на мастере — печатается и на 40 мм
    bangs = sp if bangs is None else smin(bangs, sp, 0.7)
P["bangs"] = bangs

# ---------- сборка ----------
names = list(P)
D = np.stack([P[n] for n in names]).astype(np.float32)
F = D.min(0)
print("поверхность…")
v, f, _, _ = marching_cubes(F, 0.0, spacing=(PITCH, PITCH, PITCH))
v = v + lo
m = trimesh.Trimesh(v, f[:, ::-1], process=True)
if m.volume < 0: m.invert()
comps = m.split(only_watertight=False)
m = max(comps, key=lambda c: c.area)
# цвет вершины = деталь с ближайшей поверхностью (минимальное поле) — трилинейно по сетке
idx = np.clip(np.round((m.vertices - lo) / PITCH).astype(int), 0, np.array(F.shape) - 1)
owner = D[:, idx[:, 0], idx[:, 1], idx[:, 2]].argmin(0)
key = lambda n: {"croc_L": "croc", "croc_R": "croc", "arm_L": "skin", "arm_R": "skin", "glove_L": "glove",
                 "glove_R": "glove", "face": "skin"}.get(n, n)
cols = np.array([COL[key(n)] for n in names], np.uint8)
vc = np.c_[cols[owner], np.full(len(owner), 255, np.uint8)]
m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=vc)
print("граней %d, замкнута %s, размер %s" % (len(m.faces), m.is_watertight, m.extents.round(1)))
m.export(os.path.join(OUT, "emm_color.glb"))
trimesh.Trimesh(m.vertices, m.faces).export(os.path.join(OUT, "emm_whole.stl"))
json.dump({"parts": names, "colors": {n: COL[key(n)] for n in names},
           "faces_per_part": {n: int((owner[m.faces[:, 0]] == i).sum()) for i, n in enumerate(names)}},
          open(os.path.join(OUT, "parts_truth.json"), "w"), ensure_ascii=False, indent=1)
print("ok", OUT)
