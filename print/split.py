"""Разрезка фигурки на цветные детали по разметке спереди + штыри/гнёзда + полость.
usage: python3 split.py model.glb annot.jpg outdir [height_mm=40]
"""
import sys, os, json
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure, color

SRC, ANNOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
H = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
P = float(os.environ.get("PITCH", 0.1))  # мм на воксель
WALL = 1.2         # стенка полых деталей
CLEAR = 0.15       # радиальный зазор штырь/гнездо
os.makedirs(OUT, exist_ok=True)
mm = lambda v: int(round(v / P))

# ---------- модель -> воксели ----------
s = trimesh.load(SRC, force="scene")
m = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m.apply_scale(H / m.extents[2])
m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
vox = m.voxelized(P)
D = 6
g = np.pad(vox.matrix, D + 2)
f = ndimage.binary_fill_holes(ndimage.binary_dilation(g, iterations=D))
solid = ndimage.binary_erosion(f, iterations=D) | (g & f)
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
NAMES = ["hair", "glasses", "head", "jacket", "shirt", "arm_right", "hand", "mug",
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
depth = (frontr[None] - xr) * P  # мм от передней поверхности
# первая сплошная полоса от фронта
first_run = np.zeros_like(solid)
rr = sol_r.copy()
cum = np.cumsum(~rr[::-1], axis=0)[::-1]  # число пустот впереди (в осях xr)
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
dext = ndimage.distance_transform_edt(solid) * P
lab3[(lab3 == ID["hair"]) & (dext > 2.5)] = ID["head"]

# обрывки -> к соседу: у каждой детали один цельный кусок
for it in range(3):
    for lid in range(1, len(NAMES) + 1):
        lb, n = ndimage.label(lab3 == lid)
        if n <= 1:
            continue
        sz = ndimage.sum(lab3 == lid, lb, range(1, n + 1))
        for j in np.argsort(sz)[::-1][1:]:
            comp = lb == (j + 1)
            ring = ndimage.binary_dilation(comp, iterations=2) & solid & ~comp
            vals = lab3[ring]; vals = vals[(vals != lid) & (vals > 0)]
            if len(vals):
                lab3[comp] = np.bincount(vals).argmax()
np.save(os.path.join(OUT, "_lab3.npy"), lab3)
if os.environ.get("LABELS_ONLY"): sys.exit(0)
parts = {n: lab3 == ID[n] for n in NAMES if (lab3 == ID[n]).sum() * P**3 > 1.0}
for n, mk in parts.items():
    print("%-11s %6.1f мм³" % (n, mk.sum() * P**3))

# ---------- стыки ----------
def bbox(mk, marg):
    w = np.nonzero(mk)
    return tuple(slice(max(a.min() - marg, 0), a.max() + marg + 1) for a in w)

def cyl(center, axis, r, t0, t1, shape):
    c = np.asarray(center, float); a = np.asarray(axis, float) / np.linalg.norm(axis)
    R = mm(max(abs(t0), abs(t1)) + r) + 2
    sl = tuple(slice(max(int(c[i]) - R, 0), min(int(c[i]) + R + 1, shape[i])) for i in range(3))
    g = np.mgrid[sl].reshape(3, -1).T - c
    t = g @ a * P
    rad = np.linalg.norm(g - np.outer(g @ a, a), axis=1) * P
    ok = (t >= t0) & (t <= t1) & (rad <= r)
    out = np.zeros(shape, bool)
    out[sl] = ok.reshape(tuple(s.stop - s.start for s in sl))
    return out

names = list(parts)
add = {n: np.zeros_like(solid) for n in names}   # сплошные добавки (анкер/втулка/штырь)
cut = {n: np.zeros_like(solid) for n in names}   # гнёзда
joints = []
offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
for i, a in enumerate(names):
    for b in names[i + 1:]:
        A, Bm = parts[a], parts[b]
        cont = np.zeros_like(solid); dirs = np.zeros(solid.shape + (3,), np.int8)
        for o in offs:
            sh = np.roll(Bm, shift=tuple(-x for x in o), axis=(0, 1, 2))
            hit = A & sh
            cont |= hit
            dirs[hit] += np.array(o, np.int8)
        area = cont.sum() * P * P
        if area < 0.3:
            continue
        if area < 8.0 or "glasses" in (a, b) or {a, b} == {"shoe_left", "shoe_right"}:
            joints.append(dict(a=a, b=b, type="клей", area=round(area, 1)))
            continue
        cidx = np.argwhere(cont)
        score = dext[tuple(cidx.T)]
        j = int(np.argmax(score))
        r = min(1.0, score[j] - 0.9)
        if r < 0.45:
            joints.append(dict(a=a, b=b, type="клей", area=round(area, 1)))
            continue
        p0 = cidx[j].astype(float)
        near = np.linalg.norm(cidx - p0, axis=1) < mm(2.5)
        nab = dirs[tuple(cidx[near].T)].sum(0).astype(float)  # из a в b
        if np.linalg.norm(nab) == 0:
            continue
        nab /= np.linalg.norm(nab)
        pa = p0 + 0.5 * nab  # точка на границе a|b
        def room(part, start, n):
            for tt in np.arange(0.2, 3.01, 0.1):
                q = np.round(start + n * tt / P).astype(int)
                if (q < 0).any() or (q >= solid.shape).any() or not parts[part][tuple(q)]:
                    return tt
            return 3.0
        ra, rb = room(a, pa, -nab), room(b, pa, nab)
        # штырь растёт из детали с меньшим запасом в деталь с большим
        if rb >= ra:
            small, big, nvec, Lroom = a, b, nab, rb
        else:
            small, big, nvec, Lroom = b, a, -nab, ra
        Lp = min(2.0, Lroom - 0.9)
        if Lp < 1.0:
            joints.append(dict(a=a, b=b, type="клей", area=round(area, 1)))
            continue
        p = pa
        peg = cyl(p, nvec, r, -0.05, Lp, solid.shape)
        anchor = cyl(p, nvec, r + 0.8, -1.5, 0.0, solid.shape) & parts[small]
        sleeve = cyl(p, nvec, r + 0.8, 0.0, Lp + 1.0, solid.shape) & parts[big]
        sock = cyl(p, nvec, r + CLEAR, -0.1, Lp + 0.2, solid.shape)
        add[small] |= peg | anchor
        add[big] |= sleeve
        cut[big] |= sock
        joints.append(dict(a=small, b=big, type="штырь", d=round(2 * r, 1), L=round(Lp, 1),
                           area=round(area, 1)))
        print("стык %s -> %s: штырь Ø%.1f×%.1f мм" % (small, big, 2 * r, Lp))

# ---------- полости и сборка деталей ----------
final = {}
for n in names:
    mk = parts[n]
    solidp = mk | add[n]
    hol = ""
    if mk.sum() * P**3 > 80:
        cav = ndimage.binary_erosion(mk, iterations=mm(WALL))
        cav &= ~ndimage.binary_dilation(add[n] | cut[n], iterations=mm(0.8))
        lb, k = ndimage.label(cav)
        if k:
            sz = ndimage.sum(cav, lb, range(1, k + 1))
            cav = lb == (1 + np.argmax(sz))
            if cav.sum() * P**3 > 20:
                # слив: вниз от нижней точки полости, подальше от штырей
                zc = np.nonzero(cav.any((0, 1)))[0][0]
                cand = np.argwhere(cav[:, :, zc: zc + 3].any(2))
                far = ndimage.distance_transform_edt(~(add[n] | cut[n]))[:, :, zc]
                sc = far[tuple(cand.T)] - 0.05 * np.linalg.norm(cand - cand.mean(0), axis=1)
                cx, cy = cand[np.argmax(sc)]
                X, Y = np.ogrid[:NX, :NY]
                disk = (X - cx) ** 2 + (Y - cy) ** 2 <= mm(0.75) ** 2
                drain = np.zeros_like(solid); drain[:, :, : zc + 1] = disk[:, :, None]
                solidp = (solidp & ~cav) & ~(drain & ~add[n])
                hol = "полая, %.0f мм³ полость" % (cav.sum() * P**3)
    solidp &= ~cut[n]
    final[n] = (solidp, hol)

COL = dict(hair=(0.35, 0.25, 0.15), glasses=(0.15, 0.15, 0.15), head=(0.95, 0.8, 0.65),
           jacket=(0.15, 0.15, 0.2), shirt=(0.9, 0.9, 0.9), arm_right=(0.15, 0.15, 0.2),
           hand=(0.95, 0.8, 0.65), mug=(0.85, 0.55, 0.2), pants=(0.15, 0.15, 0.2),
           shoe_left=(0.05, 0.05, 0.05), shoe_right=(0.05, 0.05, 0.05))
info = []
for n, (mk, hol) in final.items():
    sl = bbox(mk, 3)
    sub = np.pad(mk[sl], 1).astype(np.float32)
    fld = ndimage.gaussian_filter(sub, 0.6)
    v, fc, _, _ = measure.marching_cubes(fld, 0.5)
    v = (v - 1 + [s.start for s in sl]) * P + origin
    t = trimesh.Trimesh(v, fc, process=True)
    if t.volume < 0:
        t.invert()
    t.export(os.path.join(OUT, f"{n}.stl"))
    info.append(dict(part=n, mm=t.extents.round(1).tolist(), vol=round(t.volume, 1),
                     watertight=bool(t.is_watertight), bodies=len(t.split(only_watertight=False)),
                     hollow=hol))
    print("%-11s %s объём %.0f мм³ тел=%d %s %s" % (n, t.extents.round(1), t.volume,
          info[-1]["bodies"], "ok" if t.is_watertight else "НЕ ЗАМКНУТА", hol))
json.dump(dict(parts=info, joints=joints), open(os.path.join(OUT, "parts.json"), "w"),
          ensure_ascii=False, indent=1)
np.save(os.path.join(OUT, "_labels_front.npy"), L2)
np.save(os.path.join(OUT, "_lab3.npy"), lab3)
