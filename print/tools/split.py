"""Шаг 1 «разборки»: нарезка фигурки на цветные детали по разметке спереди.
usage: python3 split.py model.glb annot.jpg outdir
Все «художественные» размеры заданы в мм для эталонной фигурки высотой 40 мм.
Сетка по умолчанию 0,04 мм на эталоне = 0,1 мм на мастер-модели 10 см.
Результат: outdir/labels.npz (метки вокселей) и outdir/master_100mm/*.stl (детали без штырей).
Шаг 2 — finalize.py: итоговый размер, штыри/гнёзда, полости.
"""
import sys, os, json
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure, color

SRC, ANNOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
H = 40.0                                     # эталонная высота для художественных размеров
P = float(os.environ.get("PITCH", 0.04))     # мм эталона на воксель (0,04 = 0,1 мм на 10 см)
MASTER = float(os.environ.get("MASTER", 100.0))
os.makedirs(OUT, exist_ok=True)
mm = lambda v: int(round(v / P))

# ---------- модель -> воксели ----------
s = trimesh.load(SRC, force="scene")
m = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m.apply_scale(H / m.extents[2])
m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
vox = m.voxelized(P)
D = max(2, int(round(0.6 / P)))
g = np.pad(vox.matrix, D + 2)
f = ndimage.binary_fill_holes(ndimage.binary_dilation(g, iterations=D))
solid = ndimage.binary_erosion(f, iterations=D) | (g & f)
del f
pad = 3
solid = np.pad(solid[D + 2:-(D + 2), D + 2:-(D + 2), D + 2:-(D + 2)], pad)
origin = vox.translation - pad * P
NX, NY, NZ = solid.shape
print("воксели", solid.shape)

# ---------- разметка ----------
im = np.asarray(Image.open(ANNOT).convert("RGB"))
ih = im.shape[0]
sil = im.min(2) < 240
sil[: int(ih * 0.13)] = False  # статус-бар телефона
sil = ndimage.binary_closing(sil, iterations=4)
lab, k = ndimage.label(sil)
sil = lab == (1 + np.argmax(ndimage.sum(sil, lab, range(1, k + 1))))
sil = ndimage.binary_fill_holes(sil)
r0, r1 = np.nonzero(sil.any(1))[0][[0, -1]]
c0, c1 = np.nonzero(sil.any(0))[0][[0, -1]]
hsv = color.rgb2hsv(im)
hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
col = sil & (sat > 0.28) & (val > 0.15)
cls = np.zeros(im.shape[:2], np.uint8)  # 0 серый/нет
cls[col & ((hue < 38) | (hue > 340))] = 1  # красный/коричневый
cls[col & (hue >= 38) & (hue < 75)] = 2    # жёлтый
cls[col & (hue >= 75) & (hue < 170)] = 3   # зелёный
cls[col & (hue >= 190) & (hue < 250)] = 4  # синий
cls[col & (hue >= 250) & (hue <= 340)] = 5 # фиолетовый
for c in range(1, 6):  # чистка шума
    mk = ndimage.binary_opening(cls == c, iterations=3)
    cls[(cls == c) & ~mk] = 0

hgt = r1 - r0
rel = lambda r: (r - r0) / hgt  # 0 верх, 1 низ
NAMES = ["hair", "glasses", "head", "jacket", "shirt", "arm_right", "hand", "mug", "eyes", "tie",
         "pants", "shoe_left", "shoe_right"]
ID = {n: i + 1 for i, n in enumerate(NAMES)}
L = np.zeros(im.shape[:2], np.uint8)
L[cls == 3] = ID["hair"]
L[cls == 5] = ID["jacket"]
L[cls == 4] = ID["pants"]
cmid = (c0 + c1) / 2
for c, fn in [(1, "red"), (2, "yellow")]:
    lb, n = ndimage.label(cls == c)
    for i in range(1, n + 1):
        rr, cc = np.nonzero(lb == i)
        if len(rr) < 300:
            continue
        y, x = rel(rr.mean()), cc.mean()
        if fn == "red":
            name = "glasses" if y < 0.45 else ("mug" if y < 0.75 else
                   ("shoe_left" if x < cmid else "shoe_right"))
        else:
            name = "head" if y < 0.45 else ("hand" if x < cmid else "arm_right")
        L[lb == i] = ID[name]
# серое между полами пиджака = рубашка
rows_j = np.nonzero((L == ID["jacket"]).any(1))[0]
for r in range(rows_j.min(), rows_j.max() + 1):
    jc = np.nonzero(L[r] == ID["jacket"])[0]
    if len(jc) < 2:
        continue
    seg = np.arange(jc.min(), jc.max() + 1)
    gap = seg[(L[r, seg] == 0) & sil[r, seg]]
    L[r, gap] = ID["shirt"]
# остальное серое -> ближайшая метка
idx = ndimage.distance_transform_edt(L == 0, return_distances=False, return_indices=True)
L = np.where(sil, L[idx[0], idx[1]], 0)
LAYER = {ID["glasses"]: 1.4, ID["shirt"]: 1.2, ID["hand"]: 6.0, ID["mug"]: 7.0}
base = L.copy()
for lid in LAYER:
    base[base == lid] = 0
bi = ndimage.distance_transform_edt(base == 0, return_distances=False, return_indices=True)
base = np.where(sil, base[bi[0], bi[1]], 0)

# ---------- совмещение вида спереди с моделью ----------
proj = solid.any(0)  # (Y,Z)
ys = np.nonzero(proj.any(1))[0]; zs = np.nonzero(proj.any(0))[0]
Y0, Y1, Z0, Z1 = ys[0], ys[-1], zs[0], zs[-1]
def sample(img, flip):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="ij")
    u = (yy - Y0) / max(Y1 - Y0, 1)
    if flip:
        u = 1 - u
    v = (Z1 - zz) / max(Z1 - Z0, 1)
    ci = np.clip(np.round(c0 + u * (c1 - c0)).astype(int), 0, im.shape[1] - 1)
    ri = np.clip(np.round(r0 + v * (r1 - r0)).astype(int), 0, ih - 1)
    return img[ri, ci]
best = None
for flip in (False, True):
    sl = sample(sil, flip)
    iou = (sl & proj).sum() / (sl | proj).sum()
    if best is None or iou > best[0]:
        best = (iou, flip)
print("совпадение силуэта IoU=%.2f" % best[0], "зеркально" if best[1] else "")
FLIP = best[1]
L2, B2 = sample(L, FLIP), sample(base, FLIP)
# камера matplotlib azim=0 смотрит с +x; при зеркале — с -x
FRONT = -1 if FLIP else 1

# ---------- метки вокселей ----------
xs = np.arange(NX)[:, None, None]
xr = xs if FRONT > 0 else (NX - 1 - xs)  # «вперёд» = рост xr
sol_r = solid if FRONT > 0 else solid[::-1]
frontr = np.where(sol_r.any(0), NX - 1 - np.argmax(sol_r[::-1], axis=0), -1)
depth = ((frontr[None] - xr) * P).astype(np.float32)  # мм от передней поверхности
# первая сплошная полоса от фронта
first_run = np.zeros_like(solid)
rr = sol_r.copy()
cum = np.cumsum(~rr[::-1], axis=0, dtype=np.int16)[::-1]  # число пустот впереди (в осях xr)
gaps_ahead = cum - (~rr)
first_run_r = rr & (gaps_ahead == (NX - 1 - frontr)[None])
first_run = first_run_r if FRONT > 0 else first_run_r[::-1]

lab3 = np.where(solid, L2[None], 0).astype(np.uint8)
for lid, cap in LAYER.items():
    sel = solid & (L2[None] == lid)
    keep = sel & (depth <= cap) & first_run
    lab3[sel & ~keep] = np.broadcast_to(B2[None], solid.shape)[sel & ~keep]
# затылок -> волосы (выше нижнего края очков, задняя половина головы)
hd = lab3 == ID["head"]
if hd.any():
    gz = np.nonzero((L2 == ID["glasses"]).any(0))[0]
    zcut = gz.min() if len(gz) else np.nonzero(hd.any((0, 1)))[0].mean()
    xh = np.nonzero(hd.any((1, 2)))[0]
    xc = (xh.min() + xh.max()) / 2
    back = (xs < xc) if FRONT > 0 else (xs > xc)
    zz = np.arange(NZ)[None, None, :]
    lab3[hd & back & (zz >= zcut)] = ID["hair"]
lab3[solid & (lab3 == 0)] = ID["jacket"]
def coarse_edt(mask, f=2):
    """расстояние до наружной поверхности на прореженной сетке (экономия памяти)"""
    c = ndimage.distance_transform_edt(mask[::f, ::f, ::f]).astype(np.float32) * (P * f)
    return c
_dc = coarse_edt(solid)
def dext_at(sel):
    w = np.nonzero(sel)
    return w, _dc[w[0] // 2, w[1] // 2, w[2] // 2]
_w, _d = dext_at(lab3 == ID["hair"])
_k = _d > 2.5
lab3[_w[0][_k], _w[1][_k], _w[2][_k]] = ID["head"]
del _w, _d, _k, _dc

# ================= чистовая доводка границ =================
NL = len(NAMES) + 1
# 1) сглаживание всех границ (кроме слоёв, которые строим заново)
sm_sig = 0.45 / P
score = np.full(solid.shape, -1.0, np.float32); best_l = lab3.copy()
for lid in range(1, NL):
    mk = lab3 == lid
    if not mk.any():
        continue
    sl = tuple(slice(max(a.min() - 8, 0), a.max() + 9) for a in np.nonzero(mk))
    g = ndimage.gaussian_filter(mk[sl].astype(np.float32), sm_sig)
    sc = score[sl]; bl = best_l[sl]
    upd = g > sc
    sc[upd] = g[upd]; bl[upd] = lid
lab3 = np.where(solid, best_l, 0).astype(np.uint8)
for lid in (ID["glasses"], ID["shirt"]):
    lab3[lab3 == lid] = 0
lab3[solid & (lab3 == 0)] = 0
# дыры после снятия слоёв -> ближайшая метка
miss = solid & (lab3 == 0)
while miss.any():  # наращиваем соседние метки внутрь дыр
    w = np.nonzero(miss)
    bb = tuple(slice(max(a.min() - 2, 0), a.max() + 3) for a in w)
    grown = ndimage.grey_dilation(lab3[bb], size=3)
    fill = miss[bb] & (grown > 0)
    if not fill.any():
        break
    lab3[bb][fill] = grown[fill]
    miss = solid & (lab3 == 0)

# 2) рельеф спереди: базовая поверхность и выступы
hr = frontr.astype(float); hasf = frontr >= 0
w = hasf.astype(float); hz = np.where(hasf, hr, 0)
basef = ndimage.gaussian_filter(hz * w, 0.8 / P) / np.maximum(ndimage.gaussian_filter(w, 0.8 / P), 1e-6)
prot = (hz - basef) * P
def vox_layer(mask2d, dmax, base_embed=None):
    """воксели первого слоя спереди в пределах mask2d (Y,Z)"""
    if base_embed is None:
        sel = (depth <= dmax)
    else:  # всё, что выше базовой поверхности минус embed
        sel = (xr >= (basef - base_embed / P)[None]) & (depth <= dmax)
    return solid & first_run & mask2d[None] & sel

# 3) очки по рельефу: компонента с «окнами» = оправа, бугорки в окнах = глаза
G0 = ndimage.binary_dilation(L2 == ID["glasses"], iterations=int(2.0 / P))
fr0 = G0 & (prot > 0.2) & hasf
fr0 = ndimage.binary_closing(fr0, iterations=max(1, int(0.2 / P)))
lb, n = ndimage.label(fr0)
best_c, best_h = None, 0
for j in range(1, n + 1):
    c = lb == j
    if c.sum() * P * P < 3:
        continue
    hole = (ndimage.binary_fill_holes(c) & ~c).sum()
    if hole > best_h:
        best_c, best_h = c, hole
fr = best_c if best_c is not None else np.zeros_like(fr0)
fr = ndimage.gaussian_filter(fr.astype(np.float32), 0.12 / P) > 0.5
lens = ndimage.binary_fill_holes(fr) & ~fr
lb, n = ndimage.label(lens)
eyes2d = np.zeros_like(fr)
yy, zz = np.mgrid[:fr.shape[0], :fr.shape[1]]
if n:
    sz = ndimage.sum(lens, lb, range(1, n + 1))
    for j in np.argsort(sz)[::-1][:2]:
        ln = lb == j + 1
        if ln.sum() * P * P < 2:
            continue
        bump = ln & (prot > 0.15)
        bl, bn = ndimage.label(bump)
        if bn:
            bs = ndimage.sum(bump, bl, range(1, bn + 1))
            bmk = bl == 1 + int(np.argmax(bs))
            cy, cz = np.argwhere(bmk).mean(0)
            rad = np.sqrt(bmk.sum() / np.pi) * 1.08
        else:
            cy, cz = np.argwhere(ln).mean(0)
            rad = 0.3 * min(np.ptp(np.nonzero(ln)[0]), np.ptp(np.nonzero(ln)[1]))
        eyes2d |= ((yy - cy) ** 2 + (zz - cz) ** 2 <= rad ** 2) & ln
gl = vox_layer(fr, 2.5, base_embed=0.35) & np.isin(lab3, [ID["head"], ID["hair"]])
ey = vox_layer(eyes2d, 0.7) & (lab3 == ID["head"])
lab3[gl] = ID["glasses"]; lab3[ey] = ID["eyes"]
print("очки: %.1f мм², глаза: %.1f мм²" % (fr.sum() * P * P, eyes2d.sum() * P * P))

# 4) ровная рубашка (V между лацканами) и галстук
S0 = L2 == ID["shirt"]
if S0.any():
    zs_ = np.nonzero(S0.any(0))[0]
    z_top, z_bot = zs_.max(), zs_.min()
    top_rows = S0[:, z_top - int(1.0 / P): z_top + 1]
    yc = np.nonzero(top_rows.any(1))[0].mean()
    Wtop = float(np.clip(np.ptp(np.nonzero(top_rows.any(1))[0]) * P, 3.2, 4.2))
    yy, zz = np.mgrid[:S0.shape[0], :S0.shape[1]]
    dy = np.abs(yy - yc) * P            # мм от оси
    zt = (z_top - zz) * P               # мм вниз от воротника
    Ltot = (z_top - z_bot) * P - 0.2
    vdepth = 0.45 * Ltot
    V = (zt >= 0) & (zt <= vdepth) & (dy <= Wtop / 2 * (1 - zt / vdepth))
    knot_h, tip_h = 1.6, 1.3
    hw = np.full(S0.shape, -1.0)
    k = (zt >= 0.15) & (zt < knot_h)
    hw[k] = (2.0 - (2.0 - 1.4) * (zt[k] - 0.15) / (knot_h - 0.15)) / 2
    b = (zt >= knot_h) & (zt < Ltot - tip_h)
    hw[b] = (1.3 + (2.7 - 1.3) * (zt[b] - knot_h) / max(Ltot - tip_h - knot_h, 1e-3)) / 2
    t = (zt >= Ltot - tip_h) & (zt <= Ltot)
    hw[t] = 2.7 / 2 * (Ltot - zt[t]) / tip_h
    TIE = dy <= hw
    # узел чуть отделим от лезвия тонкой «складкой» — за счёт ширины уже видно
    allow = np.isin(lab3, [ID["jacket"], ID["shirt"], ID["tie"]])
    old_shirt = lab3 == ID["shirt"]
    lab3[old_shirt] = ID["jacket"]
    tv = vox_layer(TIE, 0.7) & allow
    sv = vox_layer(V, 1.2) & allow & ~tv
    lab3[sv] = ID["shirt"]; lab3[tv] = ID["tie"]
    print("галстук: длина %.1f мм, рубашка V: %.1f×%.1f мм" % (Ltot, Wtop, vdepth))

# обрывки -> к соседу: у каждой детали один цельный кусок
for it in range(3):
    for lid in range(1, len(NAMES) + 1):
        lb, n = ndimage.label(lab3 == lid)
        if n <= 1:
            continue
        sz = ndimage.sum(lab3 == lid, lb, range(1, n + 1))
        allobj = ndimage.find_objects(lb)
        for j in np.argsort(sz)[::-1][(2 if NAMES[lid - 1] == "eyes" else 1):]:
            bb = tuple(slice(max(o.start - 3, 0), o.stop + 3) for o in allobj[j])
            comp = lb[bb] == (j + 1)
            ring = ndimage.binary_dilation(comp, iterations=2) & solid[bb] & ~comp
            vals = lab3[bb][ring]; vals = vals[(vals != lid) & (vals > 0)]
            if len(vals):
                lab3[bb][comp] = np.bincount(vals).argmax()
np.savez_compressed(os.path.join(OUT, "labels.npz"), lab=lab3, origin=origin, pitch=P,
                    ref_height=H, names=np.array(NAMES))
np.save(os.path.join(OUT, "_lab3.npy"), lab3)
if os.environ.get("LABELS_ONLY"):
    sys.exit(0)

# ---------- мастер-детали 10 см (без штырей и полостей) ----------
k = MASTER / H
md = os.path.join(OUT, "master_%dmm" % MASTER)
os.makedirs(md, exist_ok=True)
for lid, n in enumerate(NAMES, 1):
    w = np.nonzero(lab3 == lid)
    if len(w[0]) * P**3 < 1.0:
        continue
    sl = tuple(slice(max(a.min() - 3, 0), a.max() + 4) for a in w)
    sub = np.pad(lab3[sl] == lid, 1).astype(np.float32)
    v, fc, _, _ = measure.marching_cubes(ndimage.gaussian_filter(sub, 0.6), 0.5)
    v = ((v - 1 + [q.start for q in sl]) * P + origin) * k
    t = trimesh.Trimesh(v, fc, process=True)
    if t.volume < 0:
        t.invert()
    t.export(os.path.join(md, n + ".stl"))
    print("мастер %-11s %s мм" % (n, t.extents.round(1)))
