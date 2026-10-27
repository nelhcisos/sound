"""Шаг 2 «разборки»: итоговый размер + штыри/гнёзда + полости.
usage: python3 finalize.py cutdir/labels.npz outdir [высота_мм=40]
Физические размеры (штырь, зазор, стенка, слив) — в мм ИТОГОВОЙ фигурки,
поэтому не уменьшаются вместе с моделью.
env: CLEAR (радиальный зазор, 0.15 смола / 0.25 FDM), WALL (стенка, 1.2), PEG (Ø штыря, 2.0),
     GLUE (детали только на клей, через запятую; по умолчанию glasses,eyes,tie,shirt)
"""
import sys, os, json
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

SRC, OUT = sys.argv[1], sys.argv[2]
HF = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
CLEAR = float(os.environ.get("CLEAR", 0.15))
WALL = float(os.environ.get("WALL", 1.2))
PEG = float(os.environ.get("PEG", 2.0))
GLUE = set(os.environ.get("GLUE", "glasses,eyes,tie,shirt").split(","))
os.makedirs(OUT, exist_ok=True)

d = np.load(SRC)
lab = d["lab"]; NAMES = [str(x) for x in d["names"]]
k = HF / float(d["ref_height"])
P = float(d["pitch"]) * k                 # мм итоговой фигурки на воксель
origin = d["origin"] * k
mm = lambda v: int(round(v / P))
solid = lab > 0
SH = lab.shape
print("итоговая высота %.0f мм, воксель %.3f мм, сетка %s" % (HF, P, SH))

ID = {n: i + 1 for i, n in enumerate(NAMES)}
present = [n for n in NAMES if (lab == ID[n]).sum() * P**3 > 0.05]
bbox = {}
for n, o in zip(NAMES, ndimage.find_objects(lab, max_label=len(NAMES))):
    if n in present and o is not None:
        bbox[n] = o

# ---------- контакты всех пар за один проход ----------
offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
pairs = {}
for o in offs:
    src = tuple(slice(max(0, -c), SH[i] - max(0, c)) for i, c in enumerate(o))
    dst = tuple(slice(max(0, c), SH[i] - max(0, -c)) for i, c in enumerate(o))
    A, B = lab[src], lab[dst]
    w = np.nonzero((A > 0) & (B > 0) & (A != B))
    if not len(w[0]):
        continue
    la, lb_ = A[w], B[w]
    pos = np.stack(w, 1) + [s.start for s in src]
    for a_, b_ in set(zip(la.tolist(), lb_.tolist())):
        if a_ > b_:
            continue  # пару берём один раз (a<b), направление из a в b
        sel = (la == a_) & (lb_ == b_)
        pairs.setdefault((a_, b_), []).append((pos[sel], np.array(o)))

dext = None
def get_dext():
    global dext
    if dext is None:
        dext = (ndimage.distance_transform_edt(solid) * P).astype(np.float32)
    return dext

def cyl(center, axis, r, t0, t1):
    """цилиндр вдоль axis от t0 до t1 мм; возвращает (рамка, маска)"""
    c = np.asarray(center, float); a = np.asarray(axis, float) / np.linalg.norm(axis)
    R = mm(max(abs(t0), abs(t1)) + r) + 2
    sl = tuple(slice(max(int(c[i]) - R, 0), min(int(c[i]) + R + 1, SH[i])) for i in range(3))
    g = np.mgrid[sl].reshape(3, -1).T - c
    t = g @ a * P
    rad = np.linalg.norm(g - np.outer(g @ a, a), axis=1) * P
    ok = (t >= t0) & (t <= t1) & (rad <= r)
    return sl, ok.reshape(tuple(s.stop - s.start for s in sl))

add = {n: [] for n in present}   # (рамка, маска) сплошных добавок
cut = {n: [] for n in present}   # (рамка, маска) гнёзд
joints = []
for (ia, ib), lst in sorted(pairs.items()):
    a, b = NAMES[ia - 1], NAMES[ib - 1]
    if a not in present or b not in present:
        continue
    cidx = np.concatenate([p for p, _ in lst])
    dirs = np.concatenate([np.repeat(o[None], len(p), 0) for p, o in lst])
    area = len(cidx) * P * P
    if area < 0.3:
        continue
    if area < 8.0 or {a, b} & GLUE or {a, b} == {"shoe_left", "shoe_right"}:
        joints.append(dict(a=a, b=b, type="клей", area=round(area, 1)))
        continue
    de = get_dext()
    score = de[tuple(cidx.T)]
    j = int(np.argmax(score))
    r = float(min(PEG / 2, score[j] - 0.9))
    if r < 0.45:
        joints.append(dict(a=a, b=b, type="клей", area=round(area, 1)))
        continue
    p0 = cidx[j].astype(float)
    near = np.linalg.norm(cidx - p0, axis=1) < mm(2.5)
    nab = dirs[near].sum(0).astype(float)
    if np.linalg.norm(nab) == 0:
        continue
    nab /= np.linalg.norm(nab)
    pa = p0 + 0.5 * nab
    def room(lid, start, n):
        for tt in np.arange(0.2, 3.01, 0.1):
            q = np.round(start + n * tt / P).astype(int)
            if (q < 0).any() or (q >= SH).any() or lab[tuple(q)] != lid:
                return tt
        return 3.0
    ra, rb = room(ia, pa, -nab), room(ib, pa, nab)
    if rb >= ra:
        small, big, nvec, Lroom = a, b, nab, rb
    else:
        small, big, nvec, Lroom = b, a, -nab, ra
    Lp = min(2.0, Lroom - 0.9)
    if Lp < 1.0:
        joints.append(dict(a=a, b=b, type="клей", area=round(area, 1)))
        continue
    sl, m_ = cyl(pa, nvec, r, -0.05, Lp); add[small].append((sl, m_))
    sl, m_ = cyl(pa, nvec, r + 0.8, -1.5, 0.0); add[small].append((sl, m_ & (lab[sl] == ID[small])))
    sl, m_ = cyl(pa, nvec, r + 0.8, 0.0, Lp + 1.0); add[big].append((sl, m_ & (lab[sl] == ID[big])))
    sl, m_ = cyl(pa, nvec, r + CLEAR, -0.1, Lp + 0.2); cut[big].append((sl, m_))
    joints.append(dict(a=small, b=big, type="штырь", d=round(2 * r, 1), L=round(Lp, 1),
                       area=round(area, 1)))
    print("стык %s -> %s: штырь Ø%.1f×%.1f мм" % (small, big, 2 * r, Lp))
dext = None

# ---------- детали: полость, слив, экспорт ----------
info = []
for n in present:
    # рамка детали с запасом под штыри
    b0 = bbox[n]
    lo = [s.start for s in b0]; hi = [s.stop for s in b0]
    for sl, _ in add[n] + cut[n]:
        lo = [min(l, s.start) for l, s in zip(lo, sl)]; hi = [max(h, s.stop) for h, s in zip(hi, sl)]
    lo = [max(l - 3, 0) for l in lo]; hi = [min(h + 3, S) for h, S in zip(hi, SH)]
    F = tuple(slice(l, h) for l, h in zip(lo, hi))
    mk = lab[F] == ID[n]
    def paint(lst):
        out = np.zeros_like(mk)
        for sl, m_ in lst:
            loc = tuple(slice(s.start - f.start, s.stop - f.start) for s, f in zip(sl, F))
            out[loc] |= m_
        return out
    ad, ct = paint(add[n]), paint(cut[n])
    part = mk | ad
    hol = ""
    if mk.sum() * P**3 > 80:
        cav = ndimage.distance_transform_edt(mk) > mm(WALL)
        if ad.any() or ct.any():
            cav &= ndimage.distance_transform_edt(~(ad | ct)) > mm(0.8)
        lb, kk = ndimage.label(cav)
        if kk:
            sz = ndimage.sum(cav, lb, range(1, kk + 1))
            cav = lb == (1 + np.argmax(sz))
            if cav.sum() * P**3 > 20:
                zc = np.nonzero(cav.any((0, 1)))[0][0]
                cand = np.argwhere(cav[:, :, zc: zc + 3].any(2))
                busy = (ad | ct)[:, :, : zc + 1].any(2)  # штыри во всей колонне под полостью
                far = ndimage.distance_transform_edt(~busy) if busy.any() else np.full(busy.shape, 1e3)
                sc = far[tuple(cand.T)] - 0.05 * np.linalg.norm(cand - cand.mean(0), axis=1)
                cx, cy = cand[np.argmax(sc)]
                X, Y = np.ogrid[:mk.shape[0], :mk.shape[1]]
                disk = (X - cx) ** 2 + (Y - cy) ** 2 <= mm(0.75) ** 2
                zcol = int(np.argmax(cav[cx, cy, :]))  # дно полости именно в этой колонне
                drain = np.zeros_like(mk); drain[:, :, : zcol + 2] = disk[:, :, None]
                part = (part & ~cav) & ~(drain & ~ad)
                # проверка: полость соединена с наружным воздухом?
                air = np.pad(~part, 1, constant_values=True)
                al, _ = ndimage.label(air)
                outside = al[0, 0, 0]
                cz = np.argwhere(cav)[0] + 1
                if al[tuple(cz)] == outside:
                    hol = "полая, %.0f мм³ полость" % (cav.sum() * P**3)
                else:  # слив не открылся — делаем деталь сплошной
                    part = (part | cav) & ~ct
                    hol = "сплошная (слив не вышел)"
    part &= ~ct
    sub = np.pad(part, 1).astype(np.float32)
    v, fc, _, _ = measure.marching_cubes(ndimage.gaussian_filter(sub, 0.6), 0.5)
    v = (v - 1 + lo) * P + origin
    t = trimesh.Trimesh(v, fc, process=True)
    if t.volume < 0:
        t.invert()
    bodies = t.split(only_watertight=False)
    big_ = max(bodies, key=lambda bb: abs(bb.volume))
    keep = [bb for bb in bodies if abs(bb.volume) > 0.05] if n == "eyes" else [big_]
    t = trimesh.util.concatenate(keep)
    t.export(os.path.join(OUT, n + ".stl"))
    info.append(dict(part=n, mm=t.extents.round(1).tolist(), vol=round(t.volume, 1),
                     watertight=bool(t.is_watertight), bodies=len(keep), hollow=hol))
    print("%-11s %s объём %.0f мм³ тел=%d %s %s" % (n, t.extents.round(1), t.volume, len(keep),
          "ok" if t.is_watertight else "НЕ ЗАМКНУТА", hol))
json.dump(dict(height=HF, voxel_mm=P, clear=CLEAR, wall=WALL, parts=info, joints=joints),
          open(os.path.join(OUT, "parts.json"), "w"), ensure_ascii=False, indent=1)
