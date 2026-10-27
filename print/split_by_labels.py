"""Разборка по объёму по меткам граней (цветным областям).
usage: python3 split_by_labels.py [--hollow] in.glb labels.npy out_dir name height_mm base_cut_mm [head_cut_mm] [config.json]
labels.npy — метка на каждую грань (после merge_vertices), см. порядок PARTS.
Каждый воксель внутри модели получает метку ближайшей поверхности,
подставка отрезается плоскостью base_cut_mm. На каждом стыке — штифт
у «родителя» и гнездо с зазором у «ребёнка».
--hollow: крупные детали полые (стенка 1,5 мм, полость только там, где деталь
толще 6 мм), штифты/гнёзда сплошные, 2 сливных отверстия Ø2,5 мм на скрытых стыках.
"""
import sys
from pathlib import Path
HOLLOW = "--hollow" in sys.argv
sys.argv = [a for a in sys.argv if a != "--hollow"]
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

src, lab_f, out, name = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
H, BASE = float(sys.argv[5]), float(sys.argv[6])
HEAD = float(sys.argv[7]) if len(sys.argv) > 7 else None  # тело выше этой высоты -> голова
PARTS = ["podstavka", "botinki", "telo", "uzel", "golova", "ruka_prav", "ruka_lev", "hvost"]
PARENT = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
HEAD_FROM, HEAD_TO = 2, 4
if len(sys.argv) > 8:  # config.json: {"parts": [...], "parent": {"child": parent}, "head_from": i, "head_to": j}
    import json
    cfg = json.load(open(sys.argv[8]))
    PARTS = cfg["parts"]
    PARENT = {int(k): v for k, v in cfg["parent"].items()}
    HEAD_FROM, HEAD_TO = cfg.get("head_from", 2), cfg.get("head_to", 4)
PITCH, GAP = 0.1, 0.15
R_MAX, L_MAX, R_MIN = 2.5, 3.0, 0.6
out.mkdir(parents=True, exist_ok=True)

s = trimesh.load(src, force="scene")
g = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
m = trimesh.Trimesh(g.vertices, g.faces, process=False)
m.merge_vertices(merge_tex=True, merge_norm=True)
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m.apply_scale(H / m.extents[2])
m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
FL = np.load(lab_f)
assert len(FL) == len(m.faces)

def compute_labels():
    # сплошное тело
    vox = m.voxelized(PITCH)
    grid = vox.matrix.copy()
    D = 6
    grid = np.pad(grid, D + 2)
    filled = ndimage.binary_fill_holes(ndimage.binary_dilation(grid, iterations=D))
    solid = ndimage.binary_erosion(filled, iterations=D) | (grid & filled)
    solid = solid[D + 2:-(D + 2), D + 2:-(D + 2), D + 2:-(D + 2)]
    org = vox.transform[:3, 3]
    print("воксели:", solid.shape, solid.sum(), flush=True)

    # метка = ближайшая поверхность
    idx = np.argwhere(solid)
    pts = idx * PITCH + org
    tree = cKDTree(m.triangles_center)
    _, nn = tree.query(pts, workers=-1)
    lab = np.full(solid.shape, -1, np.int8)
    lab[tuple(idx.T)] = FL[nn]
    print('ближайшие поверхности найдены', flush=True)
    zc = int(round((BASE - org[2]) / PITCH))
    lab[:, :, :zc][solid[:, :, :zc]] = 0
    lab[:, :, zc:][(lab[:, :, zc:] == 0)] = 1

    # сглаживание границ (мода в окне 3x3x3, 2 прохода) и уборка обрывков
    for _ in range(2):
        cnt = np.stack([ndimage.uniform_filter((lab == p).astype(np.float32), 3) for p in range(len(PARTS))])
        new = cnt.argmax(0).astype(np.int8)
        keep = solid & (np.arange(lab.shape[2]) >= zc)[None, None, :]
        lab[keep] = new[keep]
        lab[:, :, zc:][(lab[:, :, zc:] == 0)] = 1
    for p in range(len(PARTS)):
        cc, k = ndimage.label(lab == p)
        if k > 1:
            sz = ndimage.sum(np.ones_like(cc), cc, range(1, k + 1))
            lab[(cc > 0) & (cc != 1 + int(np.argmax(sz)))] = -2
    bad = lab == -2
    if bad.any():
        good = solid & ~bad
        _, ind = ndimage.distance_transform_edt(~good, return_indices=True)
        lab[bad] = lab[tuple(i[bad] for i in ind)]
    lab[~solid] = -1
    print('метки готовы', flush=True)
    return lab, solid, org, zc

CACHE = out / 'labels_cache.npz'
if CACHE.exists():
    z_ = np.load(CACHE); lab, solid, org, zc = z_['lab'], z_['solid'], z_['org'], int(z_['zc'])
    print('разметка из кэша', flush=True)
else:
    lab, solid, org, zc = compute_labels()
    np.savez(CACHE, lab=lab, solid=solid, org=org, zc=zc)
if HEAD is not None:
    zh = int(round((HEAD - org[2]) / PITCH))
    t_ = lab[:, :, zh:]; t_[t_ == HEAD_FROM] = HEAD_TO
    # оторванные от тела куски головы ниже плоскости остаются головой; убираем обрывки тела
    cc, k = ndimage.label(lab == HEAD_FROM)
    if k > 1:
        sz = ndimage.sum(np.ones_like(cc), cc, range(1, k + 1))
        lab[(cc > 0) & (cc != 1 + int(np.argmax(sz)))] = HEAD_TO
    print('голова отрезана плоскостью', HEAD, flush=True)

# штифты (считаем в окне вокруг стыка)
keep = np.zeros(lab.shape, bool)  # зоны вокруг штифтов/гнёзд — без полостей и сливов
for ch, par in PARENT.items():
    cm_full = lab == ch
    if not cm_full.any():
        continue
    face_full = cm_full & ndimage.binary_dilation(lab == par)
    fpts = np.argwhere(face_full)
    if len(fpts) < 10:
        print(PARTS[ch], ": стыка нет", flush=True); continue
    M = int((L_MAX + R_MAX + 2) / PITCH)
    lo = np.maximum(fpts.min(0) - M, 0); hi = np.minimum(fpts.max(0) + M + 1, lab.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    W = lab[sl]  # view
    cm, pm = W == ch, W == par
    fp = fpts - lo
    mu = fp.mean(0)
    n = np.linalg.svd(fp - mu, full_matrices=False)[2][-1]
    if np.dot(np.argwhere(cm).mean(0) - mu, n) < 0:
        n = -n
    face = np.zeros_like(cm); face[tuple(fp.T)] = True
    # глубина точки стыка = расстояние до внешней поверхности
    dall = ndimage.distance_transform_edt(cm | pm)
    score = dall[tuple(fp.T)]
    best = fp[np.argmax(score)].astype(float)
    r = min(R_MAX / PITCH, score.max() * 0.6)
    L = L_MAX / PITCH
    wall = 0.8 / PITCH
    I = np.indices(W.shape, sparse=True)
    rel = [I[0] - best[0], I[1] - best[1], I[2] - best[2]]
    t = rel[0] * n[0] + rel[1] * n[1] + rel[2] * n[2]
    d2 = rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 - t ** 2
    inside = cm | pm
    while r >= R_MIN / PITCH:
        outer = (t > 0) & (t < L + wall) & (d2 < (r + GAP / PITCH + wall) ** 2)
        if inside[outer].mean() > 0.97:
            break
        r *= 0.85; L = max(L * 0.9, 1.5 / PITCH)
    peg = (t > -1) & (t < L) & (d2 < r * r)
    hole = (t > -1) & (t < L + GAP / PITCH) & (d2 < (r + GAP / PITCH) ** 2)
    W[hole & cm] = -1
    W[peg] = par
    rk = r + (GAP + 1.5) / PITCH
    keep[sl] |= (t > -4.5 / PITCH) & (t < L + (GAP + 1.5) / PITCH) & (d2 < rk * rk)
    print(f"{PARTS[ch]} -> {PARTS[par]}: штифт Ø{2 * r * PITCH:.1f} × {L * PITCH:.1f} мм", flush=True)

WALL, CORE, DR = 1.5, 3.0, 1.25  # мм


def drill(W, p, a, b, r):
    """Сверлит цилиндр радиуса r (вокс.) от a к b, с выходом наружу за a; убирает только деталь p."""
    a = a.astype(float); b = b.astype(float)
    d = b - a; ln = np.linalg.norm(d)
    if ln < 1e-6:
        return
    u = d / ln; a2 = a - u * 4
    lo = np.maximum(np.floor(np.minimum(a2, b) - r - 1).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(a2, b) + r + 2).astype(int), W.shape)
    I_ = np.indices(hi - lo).reshape(3, -1).T + lo
    rel = I_ - a2
    t_ = np.clip(rel @ u, 0, ln + 4)
    sel = I_[np.linalg.norm(rel - np.outer(t_, u), axis=1) <= r]
    m_ = W[tuple(sel.T)] == p
    W[tuple(sel[m_].T)] = -1


if HOLLOW:
    for p, nm in enumerate(PARTS):
        idx = np.argwhere(lab == p)
        if not len(idx):
            continue
        lo = np.maximum(idx.min(0) - 3, 0); hi = np.minimum(idx.max(0) + 4, lab.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        W, K = lab[sl], keep[sl]
        mk = W == p
        core = ndimage.distance_transform_edt(mk) > CORE / PITCH
        if not core.any():
            continue
        cav = (ndimage.distance_transform_edt(~core) <= (CORE - WALL) / PITCH) & mk & ~K
        cc, k = ndimage.label(cav)
        other = (W >= 0) & (W != p)
        iface = mk & ndimage.binary_dilation(other) & ~K
        cand = np.argwhere(iface)
        if not len(cand):  # нет стыков — сливаем вниз
            cand = idx[idx[:, 2] <= idx[:, 2].min() + 1] - lo
        total = 0.0; holes = 0
        for j in range(1, k + 1):
            C = cc == j
            vol = C.sum() * PITCH ** 3
            if vol < 20:
                continue
            dC, ind = ndimage.distance_transform_edt(~C, return_indices=True)
            dd = dC[tuple(cand.T)]
            first = cand[np.argmin(dd)]
            picks = [first]
            far = np.linalg.norm(cand - first, axis=1) > 6 / PITCH
            if far.any():
                sub = cand[far]; picks.append(sub[np.argmin(dC[tuple(sub.T)])])
            W[C] = -1
            for a in picks:
                b = np.array([ind[i][tuple(a)] for i in range(3)])
                drill(W, p, a, b, DR / PITCH)
            total += vol; holes += len(picks)
        if total:
            print(f"{nm}: полость {total:.0f} мм³, сливов {holes}", flush=True)

for p, nm in enumerate(PARTS):
    mask = lab == p
    if not mask.any():
        continue
    f = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), 0.7)
    v, fc, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 2) * PITCH + org
    mesh = trimesh.Trimesh(v, fc)
    parts_ = mesh.split(only_watertight=False)
    if len(parts_) > 1:  # убираем крошки (оболочки полостей не трогаем — они крупные)
        mesh = trimesh.util.concatenate([q for q in parts_ if abs(q.volume) > 1.0])
    if mesh.volume < 0:
        mesh.invert()
    fn = out / f"{name}_{p + 1}_{nm}.stl"
    mesh.export(fn)
    print(fn.name, mesh.extents.round(1), "замкнута:", mesh.is_watertight, "оболочек:", len(mesh.split(only_watertight=False)), "смола %.2f см³" % (mesh.volume / 1000))
