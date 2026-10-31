"""Анаморфная шкатулка + двухслойная рваная картина (по мотивам The Witness).
Сцена (мм): x вправо(-x на экране справа, см. ниже), y вверх, z — вглубь от глаза.
Глаз E=(0, EY, -D). Передний слой картины z=0..T, задний z=T+G..2T+G.
"""
import numpy as np, manifold3d as M, trimesh, json, sys
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import find_contours

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
import os; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

D, EY = 75.0, 28.0
T, G = 1.2, 12.0
ZB1 = T + G                 # задний слой начинается
FIG_H = 26.0                # высота фигуры в плоскости картины
H = 17.5                    # полуразмер ядра шкатулки (35 мм)
RELIEF = 0.6                # высота белого рельефа
ZC = 52.0
Y0, Y1 = 8.0, 48.0          # пояс рельефа
RES = 0.05

# ---------- плоскость картины: растр ----------
xs = np.arange(-32, 32, RES) + RES / 2
ys = np.arange(0, 62, RES) + RES / 2
PX, PY = np.meshgrid(xs, ys)          # строка=y, столбец=x

f = np.load('fig_mask.npz'); fap, fwh = f['ap'], f['white']
r0, c0 = np.where(fap); py0, py1, px0, px1 = r0.min(), r0.max(), c0.min(), c0.max()
s = FIG_H / (py1 - py0); cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
def fig_lookup(x, y, m):
    # x мира: зритель смотрит вдоль +z, поэтому +x на экране слева -> зеркалим
    px = np.round(cx - x / s).astype(int); py = np.round(cy - (y - EY) / s).astype(int)
    ok = (px >= 0) & (px < m.shape[1]) & (py >= 0) & (py < m.shape[0])
    out = np.zeros(x.shape, bool); out[ok] = m[py[ok], px[ok]]; return out

S = fig_lookup(PX, PY, fap)
S = ndi.binary_fill_holes(ndi.binary_closing(S, iterations=2))
S = ndi.binary_opening(S, iterations=2)

def smooth_noise(scale_mm, seed):
    r = np.random.default_rng(seed).standard_normal(PX.shape)
    n = ndi.gaussian_filter(r, scale_mm / RES); return n / n.std()

dist = ndi.distance_transform_edt(~S) * RES
R = 1.2 + 3.5 * np.clip(0.5 + 0.35 * smooth_noise(3.0, 1), 0, 1) + 0.4 * smooth_noise(0.4, 2)
U = (dist < R)
N = smooth_noise(2.5, 3)
GAP = 0.25
A1 = U & ~S & (N > GAP); A2 = U & ~S & (N < -GAP)
# плюс по паре отдельных «прорех» в каждом слое, не совпадающих в проекции
def blob(cx_, cy_, r, seed):
    d = np.hypot(PX - cx_, PY - cy_) / r + 0.35 * smooth_noise(0.8, seed)
    return d < 1
A1 |= blob(-19, 45, 3.0, 11) | blob(17, 12, 2.2, 12)
A2 |= blob(18, 44, 2.6, 13) | blob(-17, 13, 2.4, 14)
A1 &= ~ndi.binary_dilation(A2 | S, iterations=4) | U
A2 &= ~ndi.binary_dilation(A1, iterations=4) | U
A2 &= ~A1

inner = (np.abs(PX) < 28.5) & (PY > 5.5) & (PY < 56.5)   # видимое поле внутри рамки
def fix_islands():
    global A1, A2
    for _ in range(6):
        ch = False
        for k in (1, 2):
            H_ = S | (A1 if k == 1 else A2)
            sol = ~H_; lab, n = ndi.label(sol)
            border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            isl = np.isin(lab, [i for i in range(1, n + 1) if i not in border])
            if isl.any():
                ch = True
                if k == 1: A1 |= isl; A2 &= ~isl
                else: A2 |= isl; A1 &= ~isl
        if not ch: break
fix_islands()
assert not (A1 & A2).any() and ((S | A1) & (S | A2) == S).all()
H1, H2 = S | A1, S | A2
print('figure bbox x', PX[S].min(), PX[S].max(), 'y', PY[S].min(), PY[S].max())

def cs_from_mask(mask, x0, y0, res, eps=0.03):
    pad = np.pad(mask.astype(float), 1)
    cs = find_contours(pad, 0.5)
    polys = [np.c_[x0 + (c[:, 1] - 1) * res, y0 + (c[:, 0] - 1) * res][::-1] for c in cs if len(c) > 4]
    return M.CrossSection(polys, M.FillRule.EvenOdd).simplify(eps)

def cone_hole(cs, za, zb):
    """Дыра, прорезанная строго по лучам из глаза (стенки на глаз)."""
    k = (za + D) / D
    c = cs.translate((0, -EY)).scale((k, k))
    kt = (zb + D) / (za + D)
    return M.Manifold.extrude(c, zb - za, 0, 0, (kt, kt)).translate((0, EY, za))

x0, y0 = xs[0], ys[0]
csH1 = cs_from_mask(H1, x0, y0, RES); csH2 = cs_from_mask(H2, x0, y0, RES)

def box(x0_, x1_, y0_, y1_, z0_, z1_):
    return M.Manifold.cube((x1_ - x0_, y1_ - y0_, z1_ - z0_)).translate((x0_, y0_, z0_))

# ---------- картина ----------
PW, PY0, PY1 = 31.0, 2.0, 58.0       # полуширина, низ, верх
WALL = 2.0; CL = 0.15
front = box(-PW, PW, PY0, PY1, 0, T)
front += box(-PW, PW, PY0, PY1, 0, ZB1 + T) - box(-PW + WALL, PW - WALL, PY0 + WALL, PY1 - WALL, -1, ZB1 + T + 1)
# полка-упор для заднего слоя (задаёт зазор G)
front += box(-PW, PW, PY0, PY1, T, ZB1) - box(-PW + WALL + 1, PW - WALL - 1, PY0 + WALL + 1, PY1 - WALL - 1, 0, ZB1 + 1)
# багет спереди: рамка на лицевой стороне
front += box(-PW, PW, PY0, PY1, -1.0, 0) - box(-PW + 3, PW - 3, PY0 + 3, PY1 - 3, -2, 1)
front += box(-PW + 0.8, PW - 0.8, PY0 + 0.8, PY1 - 0.8, -1.6, -1.0) - box(-PW + 2.2, PW - 2.2, PY0 + 2.2, PY1 - 2.2, -2, 1)
# ножки в паз подставки
FOOT = [(-22, -14), (14, 22)]
for a, b in FOOT: front += box(a, b, PY0 - 3.0, PY0 + 0.5, 0, ZB1 + T)
front -= cone_hole(csH1, -2.0, T + 0.3)
back = box(-PW + WALL + CL, PW - WALL - CL, PY0 + WALL + CL, PY1 - WALL - CL, ZB1, ZB1 + T)
back -= cone_hole(csH2, ZB1 - 0.3, ZB1 + T + 0.3)

# ---------- шкатулка (повернута углом к глазу) ----------
r2 = np.sqrt(2)
def rot45(m_):  # вокруг вертикали через центр
    return m_.rotate((0, 45, 0)).translate((0, 0, ZC))
def sq(h, ya, yb):
    return M.Manifold.cube((2 * h, yb - ya, 2 * h)).translate((-h, ya, -h))
shell = sq(23, 0, 4) + sq(21, 4, Y0) + sq(H, Y0, Y1) + sq(21, Y1, 51) + sq(23, 51, 54) + sq(14, 54, 60)
shell = shell + sq(10, 60, 62)
shell -= sq(H - 1.5, -1, 50.0)           # открытое дно, стенка 1.5
shell -= sq(19, -1, 2.5)                  # юбка плинтуса без лишней смолы
shell = rot45(shell)

# текстуры-камуфляж с фото рельефа
def tex(fname, box_, seed):
    im = Image.open(fname).convert('L').crop(box_)
    a = ndi.gaussian_filter(np.asarray(im).astype(float), 2)
    thr = np.percentile(a, 55) if 'relief2' in fname else np.percentile(a, 45)
    return a > thr
TEX = [tex('relief2.png', (270, 140, 1110, 1100), 0), tex('relief2.png', (700, 140, 1540, 1100), 1),
       tex('relief3.png', (235, 120, 760, 840), 2), tex('relief3.png', (790, 120, 1290, 840), 3)]

NU = int(2 * H / RES); NV = int((Y1 - Y0) / RES)
uu = (np.arange(NU) + 0.5) * RES; vv = Y0 + (np.arange(NV) + 0.5) * RES
UU, VV = np.meshgrid(uu, vv)
ang = np.deg2rad(45)
def corner(k):  # углы ядра в плане (x,z): перед, право(-x экрана?), зад, лево
    return np.array([[0, ZC - H * r2], [H * r2, ZC], [0, ZC + H * r2], [-H * r2, ZC]])[k % 4]
faces = [(3, 0), (0, 1), (1, 2), (2, 3)]   # грани: передняя-(+x), передняя-(-x)... обход
relief_all = None; cover = np.zeros_like(S)
for fi, (ka, kb) in enumerate(faces):
    A_, B_ = corner(ka), corner(kb)
    t = (B_ - A_) / np.linalg.norm(B_ - A_)
    n = np.array([t[1], -t[0]])            # нормаль наружу
    if np.dot(n, (A_ + B_) / 2 - np.array([0, ZC])) < 0: n = -n
    tx = TEX[fi]; th, tw = tx.shape
    ti = np.clip(((Y1 - VV) / (Y1 - Y0) * (th - 1)).astype(int), 0, th - 1)
    tj = np.clip((UU / (2 * H) * (tw - 1)).astype(int), 0, tw - 1)
    W = tx[ti, tj].copy()
    # точка грани -> луч из глаза -> плоскость картины z=0
    P = A_[None, None, :] + UU[..., None] * t + (RELIEF / 2) * n
    zf = P[..., 1]
    k = D / (zf + D)
    qx = P[..., 0] * k; qy = EY + (VV - EY) * k
    visible = np.dot(n, np.array([0, -D - ZC])) > 0  # грань смотрит на глаз
    if visible:
        inS = fig_lookup(qx, qy, np.ones_like(fap)) & \
              (np.interp(qy, ys, np.arange(len(ys))) >= 0)
        # маска S в плоскости картины
        ii = np.clip(np.round((qy - y0) / RES).astype(int), 0, len(ys) - 1)
        jj = np.clip(np.round((qx - x0) / RES).astype(int), 0, len(xs) - 1)
        inS = S[ii, jj]
        figW = fig_lookup(qx, qy, fwh)
        W = np.where(inS, figW, W)
        cover[ii, jj] = True
    W = ndi.binary_opening(ndi.binary_closing(W, iterations=2), iterations=2)
    cs = cs_from_mask(W, 0, Y0, RES, 0.03)
    rel = M.Manifold.extrude(cs, RELIEF + 0.3).translate((0, 0, -0.3))   # u->x, v->y, наружу->z
    Mx = np.array([[t[0], 0, n[0], A_[0]], [0, 1, 0, 0], [t[1], 0, n[1], A_[1]]])
    rel = rel.transform(Mx)
    relief_all = rel if relief_all is None else relief_all + rel
cover = ndi.binary_closing(cover, iterations=3)
miss = S & ~cover
print('пикселей фигуры мимо рельефа:', int(miss.sum()), 'из', int(S.sum()))
box_full = shell + relief_all

# ---------- подставка и глазок ----------
BZ0, BZ1 = -4.0, 88.0
base = box(-36, 36, -5, 0, BZ0, BZ1)
for a, b in FOOT: base -= box(a - CL, b + CL, -3.2, 1, -CL, ZB1 + T + CL)
pl = rot45(sq(23 + CL, -1.2, 1))
base -= pl
PEG = 1.0; PEGL = 2.0
for px_ in (-5, 5):
    base -= M.Manifold.cylinder(PEGL + 0.3, PEG + CL, PEG + CL, 32).translate((px_, -2.5, BZ0))
arm = box(-8, 8, -5, 0, -D - 1.5, BZ0)
post = box(-10, 10, 0, 38, -D - 0.8, -D + 0.8) + box(-10, 10, -5, 0, -D - 4, -D + 1)
post -= M.Manifold.cylinder(3, 1.2, 1.2, 48).translate((0, EY, -D - 1.5))
arm += post
for px_ in (-5, 5):
    arm += M.Manifold.cylinder(PEGL + 0.4, PEG, PEG, 32).translate((px_, -2.5, BZ0 - 0.4))
# ребро жёсткости под стойкой
arm += box(-1, 1, 0, 10, -D + 0.8, -D + 8)

parts = {'painting_front': front, 'painting_back': back, 'box': box_full,
         'base': base, 'eyepiece_arm': arm}
def tm(m_):
    mg = m_.to_mesh(); return trimesh.Trimesh(np.asarray(mg.vert_properties)[:, :3], np.asarray(mg.tri_verts), process=False)
info = {}
for k_, m_ in parts.items():
    t_ = tm(m_)
    t_.export(f'{OUT}/scene_{k_}.stl')
    info[k_] = dict(watertight=bool(t_.is_watertight), vol=round(float(t_.volume) / 1000, 2),
                    bodies=len(m_.decompose()), ext=[round(float(e), 1) for e in t_.extents])
# для рендера: белый рельеф отдельно
tm(rot45(sq(H, Y0, Y1)) + shell).export(f'{OUT}/render_box_beige.stl')
tm(relief_all).export(f'{OUT}/render_box_white.stl')
json.dump(dict(D=D, EY=EY, info=info), open(f'{OUT}/info.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
Image.fromarray((np.dstack([S, A1, A2]) * 255).astype(np.uint8)[::-1]).save(f'{OUT}/holes_plane.png')
np.savez_compressed(f'{OUT}/plane.npz', S=S, A1=A1, A2=A2)
