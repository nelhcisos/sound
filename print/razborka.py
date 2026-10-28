"""Разборка модели на детали: гладкие разрезы, точные булевы операции (manifold3d).

usage: python3 razborka.py config.json [--final]

Без --final: только разрезы + превью (для согласования).
С --final: штифты, полости (если "hollow": true), STL каждой детали и общий STL на столе.

config.json:
{
  "glb": "model.glb", "labels": "L.npy",      # метка цветной области на каждую грань (после merge_vertices)
  "height": 60, "out": "dir", "name": "tiger_red",
  "parts": ["podstavka", "botinki", ...], "root": 2,   # деталь, которой сначала принадлежит вся модель
  "cuts": [                                  # по порядку; child вырезается из from
    {"child": 0, "from": 2, "type": "plane", "point": [0,0,4.5], "normal": [0,0,-1]},  # normal — в сторону child
    {"child": 1, "from": 2, "type": "color"}   # гладкая поверхность через границу цвета child|все соседние цвета
                                                # (можно сузить: "against": [2])
  ],
  "pegs": {"1": 0, ...},                     # child: parent — штифт на parent, гнездо в child
  "hollow": true
}
Внешняя поверхность — исходная сетка модели (без вокселей). Разрез «color» — тонкая пластина
(thin-plate spline) через линию перехода цвета, «plane» — плоскость. Штифт ставится в самое
толстое место ровной площадки стыка, с запасом материала ≥ 1,5 мм вокруг и сплошным корнем.
"""
import sys, json
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as m3
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree, Delaunay
from skimage import measure, draw

cfg = json.load(open(sys.argv[1]))
FINAL = "--final" in sys.argv
OUT = Path(cfg["out"]); OUT.mkdir(parents=True, exist_ok=True)
PARTS = cfg["parts"]; NAME = cfg["name"]
GAP, R_MAX, L_MAX, ROOT, MARGIN = 0.15, 2.5, 3.0, 2.0, 1.5      # мм
WALL, CORE, DRAIN_R = 1.5, 3.0, 1.25
import resource
log = lambda *a: print(*a, "[%.1f ГБ]" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576), flush=True)

# ---------- модель ----------
s = trimesh.load(cfg["glb"], force="scene")
g = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
m = trimesh.Trimesh(g.vertices, g.faces, process=False)
m.merge_vertices(merge_tex=True, merge_norm=True)
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m.apply_scale(cfg["height"] / m.extents[2])
m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
FL = np.load(cfg["labels"]); assert len(FL) == len(m.faces)
cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)))
keep = np.zeros(len(m.faces), bool); keep[max(cc, key=len)] = True
m.update_faces(keep); m.remove_unreferenced_vertices(); FL = FL[keep]
if not m.is_watertight:
    n0 = len(m.faces); trimesh.repair.fill_holes(m)
    if len(m.faces) > n0:  # метки новых граней — по ближайшей старой
        t = cKDTree(m.triangles_center[:n0]); _, j = t.query(m.triangles_center[n0:])
        FL = np.r_[FL, FL[j]]
if m.volume < 0:
    m.invert()
assert m.is_watertight, "модель не замкнута"
log("модель:", len(m.faces), "граней, размер", m.extents.round(1))

M0 = m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                         tri_verts=np.asarray(m.faces, np.uint32),
                         face_id=np.arange(len(m.faces), dtype=np.uint32)))
assert M0.status() == m3.Error.NoError, M0.status()
MODEL_ID = int(np.asarray(M0.to_mesh().run_original_id)[0])


def to_tm(M):
    """Manifold -> (trimesh, метка каждой грани: цвет модели или -1000-id разреза)."""
    mg = M.to_mesh()
    v = np.asarray(mg.vert_properties)[:, :3]; f = np.asarray(mg.tri_verts)
    fid = np.asarray(mg.face_id); ri = np.asarray(mg.run_index) // 3; rid = np.asarray(mg.run_original_id)
    lab = np.full(len(f), -1, np.int64)
    for k in range(len(rid)):
        a, b = ri[k], ri[k + 1]
        lab[a:b] = FL[fid[a:b]] if rid[k] == MODEL_ID else -1000 - rid[k]
    return trimesh.Trimesh(v, f, process=False), lab


def from_tm(tm):
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                               tri_verts=np.asarray(tm.faces, np.uint32)))


def frame(n):
    n = n / np.linalg.norm(n)
    a = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    return e1, e2, n


def to_world(e1, e2, n, o):
    return np.c_[e1, e2, n, o]  # 3x4


# ---------- резаки ----------
def heightfield_cutter(o, e1, e2, n, uv_lo, uv_hi, hfun, far, step=0.2):
    """Замкнутое тело: снизу поверхность w=h(u,v), сверху w=far (сторона child)."""
    us = np.arange(uv_lo[0], uv_hi[0] + step, step); vs = np.arange(uv_lo[1], uv_hi[1] + step, step)
    U, V = np.meshgrid(us, vs, indexing="ij"); Hh = hfun(U, V)
    nu, nv = U.shape
    idx = lambda i, j, top: (top * nu * nv) + i * nv + j
    P = np.r_[np.c_[U.ravel(), V.ravel(), Hh.ravel()], np.c_[U.ravel(), V.ravel(), np.full(U.size, far)]]
    F = []
    for top in (0, 1):
        i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij"); i = i.ravel(); j = j.ravel()
        a, b, c, d = idx(i, j, top), idx(i + 1, j, top), idx(i + 1, j + 1, top), idx(i, j + 1, top)
        t1 = np.c_[a, b, c]; t2 = np.c_[a, c, d]
        if top == 0:
            t1 = t1[:, ::-1]; t2 = t2[:, ::-1]
        F += [t1, t2]
    ring = [(i, 0) for i in range(nu)] + [(nu - 1, j) for j in range(1, nv)] + \
           [(i, nv - 1) for i in range(nu - 2, -1, -1)] + [(0, j) for j in range(nv - 2, 0, -1)]
    side = []
    for k in range(len(ring)):
        (i0, j0), (i1, j1) = ring[k], ring[(k + 1) % len(ring)]
        a, b, c, d = idx(i0, j0, 0), idx(i1, j1, 0), idx(i1, j1, 1), idx(i0, j0, 1)
        side += [[a, b, c], [a, c, d]]
    F = np.vstack(F + [np.array(side)])
    W = P[:, :1] * e1 + P[:, 1:2] * e2 + P[:, 2:3] * n + o
    tm = trimesh.Trimesh(W, F, process=False)
    if tm.volume < 0:
        tm.invert()
    return from_tm(tm)


def color_membrane(cut, tm_from, lab_from):
    """Гладкая поверхность через границу цвета child|against на текущей детали from."""
    child = cut["child"]
    against = cut.get("against") or [int(x) for x in np.unique(lab_from) if x >= 0 and x != child]
    adj = tm_from.face_adjacency
    la, lb = lab_from[adj[:, 0]], lab_from[adj[:, 1]]
    sel = ((la == child) & np.isin(lb, against)) | ((lb == child) & np.isin(la, against))
    if sel.sum() < 10:
        raise RuntimeError(f"нет границы цвета для {PARTS[child]}")
    ed = tm_from.face_adjacency_edges[sel]
    pts = tm_from.vertices[ed].mean(1)
    o = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - o, full_matrices=False)
    n = vt[2]
    cf = lab_from == child
    cpos = (tm_from.triangles_center[cf] * tm_from.area_faces[cf, None]).sum(0) / tm_from.area_faces[cf].sum()
    if np.dot(cpos - o, n) < 0:
        n = -n
    e1, e2, n = frame(n)
    q = pts - o; uv = np.c_[q @ e1, q @ e2]; w = q @ n
    rng = np.random.default_rng(0)
    if len(uv) > 2500:
        k = rng.choice(len(uv), 2500, replace=False); uv, w = uv[k], w[k]
    # грани child и грани против (рядом с child): цвет должен оказаться по свою сторону поверхности
    tcen = tm_from.triangles_center
    cC = tcen[cf] - o
    af = np.isin(lab_from, against)
    near = cKDTree(tcen[cf]).query(tcen[af], distance_upper_bound=4.0)[0] < 4.0
    cA = tcen[af][near] - o
    uvC, wC = np.c_[cC @ e1, cC @ e2], cC @ n
    uvA, wA = np.c_[cA @ e1, cA @ e2], cA @ n
    if len(uvC) > 40000:
        k = rng.choice(len(uvC), 40000, replace=False); uvC, wC = uvC[k], wC[k]
    if len(uvA) > 40000:
        k = rng.choice(len(uvA), 40000, replace=False); uvA, wA = uvA[k], wA[k]
    MG = 0.35
    cu_, cw_ = uv.copy(), w.copy()
    for it in range(4):
        rbf = RBFInterpolator(cu_, cw_, kernel="thin_plate_spline", smoothing=len(cu_) * 0.02)
        hA = rbf(uvA); hC = rbf(uvC)
        badA = wA > hA - MG; badC = wC < hC + MG   # тело на стороне child / child на стороне тела
        nb = badA.sum() + badC.sum()
        if it == 0:
            log(f"  {PARTS[child]}: не по своей стороне граней: тело {badA.sum()}, деталь {badC.sum()}")
        if nb == 0:
            break
        addu = np.r_[uvA[badA], uvC[badC]]; addw = np.r_[wA[badA] + MG, wC[badC] - MG]
        if len(addu) > 1500:
            k = rng.choice(len(addu), 1500, replace=False); addu, addw = addu[k], addw[k]
        cu_ = np.r_[cu_, addu]; cw_ = np.r_[cw_, addw]
    log(f"  {PARTS[child]}: после подгонки не по своей стороне: тело {int(badA.sum())}, деталь {int(badC.sum())}")
    rbf = RBFInterpolator(cu_, cw_, kernel="thin_plate_spline", smoothing=len(cu_) * 0.02)
    hull = Delaunay(cu_); lim = max(np.abs(cw_).max(), np.abs(w).max()) + 1.0
    def hfun(U, V):
        X = np.c_[U.ravel(), V.ravel()]
        h = rbf(X)
        inside = hull.find_simplex(X) >= 0
        if (~inside).any():
            d, _ = cKDTree(cu_).query(X[~inside])
            h[~inside] *= np.exp(-(d / 3.0) ** 2)  # вне контура — плавно к плоскости
        return np.clip(h, -lim, lim).reshape(U.shape)
    # окно: проекция всей детали child (по её цветным граням) + запас
    cp = tm_from.vertices[np.unique(tm_from.faces[cf])] - o
    cu = np.c_[cp @ e1, cp @ e2]; cw = cp @ n
    lo = np.minimum(cu.min(0), uv.min(0)) - 3; hi = np.maximum(cu.max(0), uv.max(0)) + 3
    far = max(cw.max(), 0) + 5
    return heightfield_cutter(o, e1, e2, n, lo, hi, hfun, far), (o, n)


def plane_cutter(cut, ext):
    o = np.array(cut["point"], float); e1, e2, n = frame(np.array(cut["normal"], float))
    R = ext
    return heightfield_cutter(o, e1, e2, n, (-R, -R), (R, R), lambda U, V: np.zeros_like(U), R, step=R), (o, n)


# ---------- разрезы ----------
parts = {cfg["root"]: M0}
cut_info = {}  # (child, from) -> id резака
ext = float(np.abs(m.bounds).max() * 2 + 10)
for cut in cfg["cuts"]:
    ch, fr = cut["child"], cut["from"]
    tmf, labf = to_tm(parts[fr])
    if cut["type"] == "plane":
        K, (o, n) = plane_cutter(cut, ext)
    else:
        K, (o, n) = color_membrane(cut, tmf, labf)
    K = K.as_original(); kid = K.original_id()
    X = parts[fr] ^ K
    sel = []
    tot = labf == ch
    tot_area = tmf.area_faces[tot].sum()
    for comp in X.decompose():
        tc, lc = to_tm(comp)
        a = tc.area_faces[lc == ch].sum()
        if a >= 0.03 * tot_area or (cut["type"] == "plane" and cut.get("all_side", False)):
            sel.append(comp)
    if not sel:
        raise RuntimeError(f"{PARTS[ch]}: не нашёл кусок")
    C = m3.Manifold.batch_boolean(sel, m3.OpType.Add) if len(sel) > 1 else sel[0]
    parts[ch] = C
    parts[fr] = parts[fr] - C
    cut_info[(ch, fr)] = -1000 - kid
    log(f"{PARTS[ch]} из {PARTS[fr]}: объём {C.volume():.0f} мм³, нормаль {np.round(n, 2)}")

# обрывки: мелкие отдельные куски прирастают к соседу, с которым у них общий разрез
for it in range(3):
    moved = False
    for p in list(parts):
        comps = parts[p].decompose()
        if len(comps) < 2:
            continue
        comps.sort(key=lambda q: -q.volume())
        main = comps[0]
        rest = [q for q in comps[1:] if q.volume() > 0.05]  # нулевые плёнки выбрасываем
        if len(rest) != len(comps) - 1:
            parts[p] = m3.Manifold.batch_boolean([main] + rest, m3.OpType.Add) if rest else main
            moved = True
        for fr_ in rest:
            if fr_.volume() > 0.3 * main.volume():
                continue
            best, bv = None, 0
            for q in parts:
                if q == p:
                    continue
                tc, _ = to_tm(fr_)
                pts = tc.vertices[::max(1, len(tc.vertices) // 300)]
                tq, _ = to_tm(parts[q])
                dist = cKDTree(tq.vertices).query(pts)[0]
                score = (dist < 0.05).sum()
                if score > bv:
                    best, bv = q, score
            if best is not None:
                parts[best] = parts[best] + fr_
                parts[p] = parts[p] - fr_
                log(f"обрывок {fr_.volume():.1f} мм³: {PARTS[p]} -> {PARTS[best]}")
                moved = True
    if not moved:
        break

cols = np.array([[235, 235, 240], [70, 70, 80], [90, 140, 230], [200, 80, 210], [250, 210, 60], [240, 130, 40],
                 [120, 200, 60], [230, 60, 60], [30, 30, 30], [255, 120, 200], [150, 60, 220], [60, 200, 200]])


def save_preview(tag):
    sc = trimesh.Scene(); cen = m.bounds.mean(0)
    for p, M in parts.items():
        tm, _ = to_tm(M)
        if len(tm.faces) > 150000:
            tm = tm.simplify_quadric_decimation(face_count=150000)
        d = tm.bounds.mean(0) - cen; d[2] *= 0.6
        tm.apply_translation(d * 0.35)
        tm.visual.face_colors = list(cols[p % len(cols)]) + [255]
        tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        sc.add_geometry(tm, node_name=PARTS[p])
    sc.export(OUT / f"{tag}.glb")
    for p, M in parts.items():  # каждая деталь отдельно (для крупных планов стыков)
        tm, _ = to_tm(M)
        if len(tm.faces) > 120000:
            tm = tm.simplify_quadric_decimation(face_count=120000)
        tm.visual.face_colors = list(cols[p % len(cols)]) + [255]
        tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        tm.export(OUT / f"{tag}_{p + 1}_{PARTS[p]}.glb")


if not FINAL:
    save_preview("plan")
    log("план сохранён")
    sys.exit(0)

# ---------- штифты ----------
pegs = []  # (parent, child, center, dir, r, L, root)
for chs, par in cfg["pegs"].items():
    ch = int(chs)
    tc, lc = to_tm(parts[ch]); tp, lp = to_tm(parts[par])
    # площадка стыка: грани разрезов у child, лежащие на поверхности parent (точное расстояние)
    pf = np.where(lp <= -1000)[0]; cf = np.where(lc <= -1000)[0]
    if len(pf) < 3 or len(cf) < 3:
        log(f"{PARTS[ch]}: нет площадки стыка с {PARTS[par]}"); continue
    sub = tp.submesh([pf], append=True)
    n_s = int(min(2e6, max(2e4, sub.area / 0.03 ** 2)))
    sp_ = trimesh.sample.sample_surface(sub, n_s, seed=0)[0]
    d = cKDTree(sp_).query(tc.triangles_center[cf])[0]
    fidx = cf[d < 0.06]
    if len(fidx) < 3:
        log(f"{PARTS[ch]}: площадка не касается {PARTS[par]}"); continue
    dirv = -(tc.face_normals[fidx] * tc.area_faces[fidx, None]).sum(0); dirv /= np.linalg.norm(dirv)
    e1, e2, _ = frame(dirv)
    o = tc.triangles_center[fidx].mean(0)
    tri = tc.vertices[tc.faces[fidx]] - o
    A = np.stack([tri @ e1, tri @ e2], -1)
    st = 0.05
    lo = A.reshape(-1, 2).min(0) - 1; hi = A.reshape(-1, 2).max(0) + 1
    shp = np.ceil((hi - lo) / st).astype(int) + 1
    img = np.zeros(shp, bool)
    for t in A:
        rr, cc_ = draw.polygon((t[:, 0] - lo[0]) / st, (t[:, 1] - lo[1]) / st, shp)
        img[rr, cc_] = True
    img = ndimage.binary_closing(img, iterations=2)
    dt = ndimage.distance_transform_edt(img) * st
    inr = dt.max()
    # кандидаты: самые толстые места площадки (через ≥1 мм), радиус уменьшаем по шагам
    order = np.argsort(dt.ravel())[::-1]
    cands = []
    for k in order[:20000]:
        if dt.flat[k] < MARGIN + 0.6:
            break
        q = np.array(np.unravel_index(k, dt.shape))
        if all(np.linalg.norm((q - c) * st) >= 1.0 for c in cands):
            cands.append(q)
        if len(cands) >= 12:
            break
    psub = tc.submesh([fidx], append=True)
    psp = trimesh.sample.sample_surface(psub, int(min(2e6, max(2e4, psub.area / 0.03 ** 2))), seed=1)[0]
    ptree = cKDTree(psp)
    def depth(M, sign, ring):
        """Толщина материала по лучу от площадки: от пересечения с площадкой до следующей стенки."""
        ds = []
        for p0 in ring:
            a_, b_ = p0 - sign * dirv * 5, p0 + sign * dirv * 30
            hits = sorted(h.distance * 35 - 5 for h in M.ray_cast(tuple(a_), tuple(b_)))
            if not hits:
                ds.append(0.0); continue
            k_ = int(np.argmin([abs(h) for h in hits]))
            ds.append(hits[k_ + 1] - hits[k_] if k_ + 1 < len(hits) else 0.0)
        return min(ds)
    log(f"  {PARTS[ch]}: площадка {len(fidx)} граней, растр {shp}, кандидатов {len(cands)}")
    found = None; why = f"площадка мала (вписанный радиус {inr:.1f} мм)"
    for q in cands:
        c0 = o + (lo[0] + q[0] * st) * e1 + (lo[1] + q[1] * st) * e2
        c0 = psp[ptree.query(c0)[1]]
        r0 = min(R_MAX, dt[tuple(q)] - MARGIN)
        for r in (r0, r0 * 0.8, r0 * 0.65):
            if r < 0.6:
                break
            ring = [c0] + [c0 + r * (np.cos(t) * e1 + np.sin(t) * e2) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
            dch = depth(parts[ch], 1, ring); dpa = depth(parts[par], -1, ring)
            L = min(L_MAX, dch - 1.2); root = min(ROOT, dpa - 0.3)
            if L >= 1.2 and root >= 0.8:
                found = (c0, r, L, root); break
            why = f"мало глубины (child {dch:.1f}, parent {dpa:.1f} мм)"
        if found:
            break
    if not found:
        log(f"{PARTS[ch]} -> {PARTS[par]}: {why} — без штифта, на клей")
        continue
    c0, r, L, root = found
    T = to_world(e1, e2, dirv, c0)
    chm = min(0.3, r * 0.3)
    peg = (m3.Manifold.cylinder(root + L - chm, r, r, 64).translate((0, 0, -root)) +
           m3.Manifold.cylinder(chm, r, r - chm, 64).translate((0, 0, L - chm))).transform(T)
    sock = m3.Manifold.cylinder(root + 0.5 + L + GAP, r + GAP, r + GAP, 64).translate((0, 0, -root - 0.5)).transform(T)
    parts[par] = parts[par] + peg
    parts[ch] = parts[ch] - sock
    pegs.append((par, ch, c0, dirv, r, L, root))
    log(f"{PARTS[ch]} -> {PARTS[par]}: штифт Ø{2 * r:.1f} × {L:.1f} мм, корень {root:.1f} мм, площадка r={inr:.1f}")

# ---------- полости ----------
if cfg.get("hollow"):
    P = 0.1
    for p in list(parts):
        M = parts[p]
        if M.volume() < 300:
            continue
        tm, lab = to_tm(M)
        vx = tm.voxelized(P).fill()
        occ = np.pad(vx.matrix, 3); org = vx.transform[:3, 3] - 3 * P
        d = ndimage.distance_transform_edt(occ) * P
        core = d > CORE
        if not core.any():
            continue
        field = ndimage.distance_transform_edt(~core) * P - (CORE - WALL)
        field = ndimage.gaussian_filter(field.astype(np.float32), 1.5)
        v, f, _, _ = measure.marching_cubes(field, 0.0)
        cav_tm = trimesh.Trimesh(v * P + org, f)
        if cav_tm.volume < 0:
            cav_tm.invert()
        cav = from_tm(cav_tm)
        for (par, ch, c0, dv, r, L, root) in pegs:
            e1, e2, _ = frame(dv); T = to_world(e1, e2, dv, c0)
            if p == par:
                cav = cav - m3.Manifold.cylinder(root + 1.8 + 0.3, r + 1.5, r + 1.5, 48).translate((0, 0, -root - 1.8)).transform(T)
            if p == ch:
                cav = cav - m3.Manifold.cylinder(L + GAP + 1.5 + 0.5, r + GAP + 1.5, r + GAP + 1.5, 48).translate((0, 0, -0.5)).transform(T)
        comps = [q for q in cav.decompose() if q.volume() > 20]
        if not comps:
            continue
        # сливы — на скрытых площадках стыков (грани от резаков), подальше от штифтов
        cand = tm.triangles_center[lab <= -1000]
        for (par, ch, c0, dv, r, L, root) in pegs:
            if p in (par, ch):
                cand = cand[np.linalg.norm(np.cross(cand - c0, dv), axis=1) > r + GAP + 1.5 + DRAIN_R]
        if len(cand) == 0:
            cand = tm.triangles_center[tm.triangles_center[:, 2] < tm.bounds[0][2] + 0.5]
        total, holes = 0, 0
        for q in comps:
            qt, _ = to_tm(q)
            dist, j = cKDTree(qt.vertices).query(cand)
            picks = [int(np.argmin(dist))]
            far = np.linalg.norm(cand - cand[picks[0]], axis=1) > 6
            if far.any():
                k2 = np.where(far)[0]; picks.append(int(k2[np.argmin(dist[k2])]))
            drills = []
            for k in picks:
                a = cand[k]; b = qt.vertices[j[k]]
                dv = b - a; ln = np.linalg.norm(dv); dv /= ln
                e1, e2, _ = frame(dv)
                drills.append(m3.Manifold.cylinder(ln + 1.5, DRAIN_R, DRAIN_R, 32).translate((0, 0, -1.0)).transform(to_world(e1, e2, dv, a)))
            M = M - q - m3.Manifold.batch_boolean(drills, m3.OpType.Add)
            total += q.volume(); holes += len(picks)
        parts[p] = M
        log(f"{PARTS[p]}: полость {total:.0f} мм³, сливов {holes}")

# ---------- экспорт ----------
outs = []
for p in sorted(parts):
    tm, _ = to_tm(parts[p])
    tm.merge_vertices()
    # плёнки нулевой толщины — выбрасываем
    cs = tm.split(only_watertight=False)
    if len(cs) > 1:
        tm = trimesh.util.concatenate([c for c in cs if abs(c.volume) > 0.5])
        tm.merge_vertices()
    # точки самокасания (ребро у 4 граней) — заделываем перемычками 0,2 мм
    for _fix in range(4):
        u_, c_ = np.unique(tm.edges_sorted, axis=0, return_counts=True)
        bad = u_[c_ != 2]
        if not len(bad):
            break
        pts = tm.vertices[bad].mean(1)
        keys = np.unique(np.round(pts / 0.3).astype(int), axis=0) * 0.3
        sph = [m3.Manifold.sphere(0.2, 12).translate(tuple(k)) for k in keys]
        parts[p] = M2 = parts[p] + m3.Manifold.batch_boolean(sph, m3.OpType.Add)
        tm, _ = to_tm(M2); tm.merge_vertices()
        cs = tm.split(only_watertight=False)
        if len(cs) > 1:
            tm = trimesh.util.concatenate([c for c in cs if abs(c.volume) > 0.5]); tm.merge_vertices()
    fn = OUT / f"{NAME}_{p + 1}_{PARTS[p]}.stl"
    tm.export(fn)
    outs.append(tm)
    log(fn.name, tm.extents.round(1), "замкнута:", tm.is_watertight, "кусков:", len(parts[p].decompose()),
        "смола %.2f см³" % (parts[p].volume() / 1000))
order = sorted(range(len(outs)), key=lambda i: -outs[i].extents[1])
x = y = rowh = 0; plate = []
for i in order:
    t = outs[i].copy(); e = t.extents
    if x + e[0] > 120:
        x = 0; y += rowh + 3; rowh = 0
    t.apply_translation([x - t.bounds[0][0], y - t.bounds[0][1], -t.bounds[0][2]])
    plate.append(t); x += e[0] + 3; rowh = max(rowh, e[1])
trimesh.util.concatenate(plate).export(OUT / f"{NAME}_all_parts.stl")
save_preview("final")
log("готово")
