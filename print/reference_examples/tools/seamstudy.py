# seamstudy.py dir pairs.txt out.tsv : угол плоскости к оси, кто в кого входит, глубина «под край», ступенька объёма
import trimesh, numpy as np, sys, os
from scipy.spatial import cKDTree
D, PAIRS, OUT = sys.argv[1:4]
cache = {}
def get(n):
    if n not in cache:
        m = trimesh.load(os.path.join(D, n + ".stl"), force="mesh")
        k = int(min(900000, max(150000, m.area * 25)))
        p, fi = trimesh.sample.sample_surface(m, k)
        cache[n] = (m, p, m.face_normals[fi], cKDTree(p))
    return cache[n]
rows = []
for line in open(PAIRS):
    line = line.strip()
    if not line or line.startswith("#"): continue
    a, b, label = (line.split("|") + [""])[:3]
    A = get(a); B = get(b)
    d, _ = B[3].query(A[1], distance_upper_bound=0.6); ca = d < 0.6
    if ca.sum() < 200: print("нет контакта", a, b); continue
    P = A[1][ca]; N = A[2][ca]
    best = None
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(N), 500):
        c = (N @ N[i] > 0.985).sum()
        if best is None or c > best[0]: best = (c, N[i])
    n = best[1]; sel = N @ n > 0.985; n = N[sel].mean(0); n /= np.linalg.norm(n); sel = N @ n > 0.985
    off = np.median(P[sel] @ n); flat = P[sel][np.abs(P[sel] @ n - off) < 0.3]
    if len(flat) < 50: print("нет плоскости", a, b); continue
    c = flat.mean(0); c = c - n * ((c @ n) - off)
    u = np.cross(n, [0, 0, 1] if abs(n[2]) < 0.9 else [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    q = np.c_[(flat - c) @ u, (flat - c) @ v]
    ext = np.percentile(q, 98, 0) - np.percentile(q, 2, 0); Rf = ext.max() / 2
    flatfrac = sel.mean()
    # кто входит: точки A за плоскостью (в сторону B) у центра = шип A; точки B по эту сторону = шип B
    sa = (A[1] - c) @ n; ra = np.linalg.norm((A[1] - c) - np.outer(sa, n), axis=1)
    sb = (B[1] - c) @ n; rb = np.linalg.norm((B[1] - c) - np.outer(sb, n), axis=1)
    tenA = ((sa > 0.4) & (ra < Rf)).sum(); tenB = ((sb < -0.4) & (rb < Rf)).sum()
    inner, outer, nIO = (a, b, n) if tenA >= tenB else (b, a, -n)
    I = cache[inner]; O = cache[outer]
    # ось внутренней детали: от центра среза к центру её точек на расстоянии 1..4 радиуса среза (0° = плоскость поперёк оси)
    si = (I[1] - c) @ nIO
    far = (si < -Rf) & (si > -4 * Rf)
    ang = np.nan; elong = np.nan
    if far.sum() > 100:
        g = I[1][far].mean(0); ax_ = (c - g) / np.linalg.norm(c - g)
        ang = np.degrees(np.arccos(min(1, abs(ax_ @ nIO))))
        elong = np.linalg.norm(c - g) / max(Rf, 1e-6)
    # «под край»: наружная деталь заходит за плоскость на сторону внутренней в радиусе Rf+5 мм от оси стыка
    so = (O[1] - c) @ nIO; ro = np.linalg.norm((O[1] - c) - np.outer(so, nIO), axis=1)
    m_ = (so < -0.2) & (ro < Rf + 5)
    over = float(np.percentile(-so[m_], 98)) if m_.sum() > 30 else 0.0
    # толщина у шва: луч из центра в каждую деталь
    def depth(M, o, dvec):
        h, _, _ = M[0].ray.intersects_location([o + dvec * 0.05], [dvec])
        return float(np.min(np.linalg.norm(h - o, axis=1))) if len(h) else 0.0
    dIn = depth(I, c, -nIO); dOut = depth(O, c, nIO)
    # ступенька: сечение наружной в 0.5 мм за плоскостью / сечение внутренней в 0.5 мм перед плоскостью
    def area(m, o, nn):
        s = m.section(plane_origin=o, plane_normal=nn)
        if s is None: return 0.0
        try:
            pl, _ = s.to_planar(); return float(pl.area)
        except Exception: return 0.0
    aO = area(O[0], c + nIO * 0.5, nIO); aI = area(I[0], c - nIO * 0.5, nIO)
    step = aO / aI if aI > 0 else np.nan
    rows.append((label or "%s-%s" % (a, b), inner, outer, round(2 * Rf, 1), round(flatfrac * 100), round(ang, 0) if ang == ang else -1,
                 round(over, 1), round(step, 2) if step == step else -1, round(dIn, 1), round(dOut, 1)))
    print("%-24s шип из %-14s в %-14s срез %5.1f мм, плоско %3d%%, наклон к оси %3s°, под край %4.1f мм, ступенька x%s, толщина у шва: шипа %s / гнезда %s" % rows[-1])
with open(OUT, "w") as f:
    f.write("стык\tшип_из\tгнездо_в\tсрез_мм\tплоско_%\tнаклон_к_оси\tпод_край_мм\tступенька\tтолщина_шип\tтолщина_гнездо\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
