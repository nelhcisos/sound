"""Шаг 3 разборки v3: итоговый размер + квадратные шипы и гнёзда (всё в мм ИТОГОВОЙ фигурки).

usage: python3 keys.py cutdir outdir [высота=40] [--fdm]
  cutdir — результат cut.py (master_100mm/*.stl + joints.json)
Параметры (env): WMIN=1.0 WMAX=8 KW=0.42 (доля меньшей стороны среза) KH=0.8 (высота/ширина)
                 WALL=0.6 (стенка вокруг гнезда) FLOOR=0.6 (дно гнезда) CS=0.10 (зазор стенок) CT=0.25 (зазор торца)
                 R=0.3 (скругление) DRAFT=4 (уклон стенок, °)
Результат: outdir/<деталь>.stl, outdir/keys.json, outdir/keys_check.png (разрезы по каждому шипу).
"""
import sys, os, json, argparse
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3lib import to_man, to_tm, frame, mat34
import manifold3d as m3
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polylabel

ap = argparse.ArgumentParser(); ap.add_argument("cutdir"); ap.add_argument("outdir")
ap.add_argument("H", nargs="?", type=float, default=40.0); ap.add_argument("--fdm", action="store_true")
a = ap.parse_args()
E = lambda k, d: float(os.environ.get(k, d))
WMIN, WMAX, KW, KH = E("WMIN", 1.0), E("WMAX", 8), E("KW", 0.42), E("KH", 0.8)
WALL, FLOOR, R, DRAFT = E("WALL", 0.6), E("FLOOR", 0.6), E("R", 0.3), E("DRAFT", 4)
CS, CT = (E("CS", 0.2), E("CT", 0.3)) if a.fdm else (E("CS", 0.10), E("CT", 0.25))
os.makedirs(a.outdir, exist_ok=True)

J = json.load(open(a.cutdir + "/joints.json"))
s = a.H / J["master_height"]
T = {}
for n in J["parts"]:
    t = trimesh.load("%s/master_100mm/%s.stl" % (a.cutdir, n), force="mesh"); t.apply_scale(s); T[n] = t
M = {n: to_man(t) for n, t in T.items()}
print("масштаб %.4f (мастер %.1f мм -> %.1f мм), зазоры: стенки %.2f, торец %.2f" % (s, J["master_height"], a.H, CS, CT))


def prism(w, h, taper_w, z0=0.0):
    """квадрат w со скруглением R, высота h, к торцу ширина taper_w; от z0 вверх"""
    r = min(R, w * 0.25)
    cs = m3.CrossSection.square([w - 2 * r, w - 2 * r], True).offset(r, m3.JoinType.Round, 2.0, 24)
    return m3.Manifold.extrude(cs, h, 0, 0.0, (taper_w / w, taper_w / w)).translate([0, 0, z0])


def owner_of(points):
    """какой детали принадлежат точки (большинство)"""
    best, cnt = None, 0
    for n, t in T.items():
        lo, hi = t.bounds
        inb = np.all((points >= lo - 0.01) & (points <= hi + 0.01), axis=1)
        if not inb.any(): continue
        c = t.contains(points[inb]).sum()
        if c > cnt: best, cnt = n, c
    return best, cnt


adds = {n: [] for n in T}; subs = {n: [] for n in T}; report = []; checks = []
tan = np.tan(np.radians(DRAFT))
# несущие стыки (внутренние, снятые позже) получают место первыми
for j in reversed(J["joints"]):
    KHj = float(j.get("kh", KH))
    part = j["part"]; n = np.asarray(j["n"]); p = np.asarray(j["p"]) * s
    u, v, w = frame(n)
    t = T[part]
    sec = t.section(plane_origin=p + n * 0.02, plane_normal=n)
    if sec is None:
        print("%s: нет сечения на плоскости стыка" % part); continue
    polys = []
    for e in sec.discrete:
        q = np.c_[(e - p) @ u, (e - p) @ v]
        if len(q) >= 4: polys.append(Polygon(q).buffer(0))
    # внешние контуры с дырами: собираем через symmetric difference
    shape = Polygon()
    for P in sorted(polys, key=lambda P: -P.area): shape = shape.symmetric_difference(P)
    geoms = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
    for g in geoms:
        if g.area < 0.5: continue
        # сосед по ту сторону плоскости
        xs, ys = g.representative_point().coords[0]
        grid = [g.representative_point()]
        mn = np.array(g.bounds)
        pts2 = []
        for gx in np.linspace(mn[0], mn[2], 9):
            for gy in np.linspace(mn[1], mn[3], 9):
                from shapely.geometry import Point
                if g.contains(Point(gx, gy)): pts2.append((gx, gy))
        pts2 = np.array(pts2[:60] or [(xs, ys)])
        P3 = p + pts2[:, :1] * u + pts2[:, 1:2] * v - n * 0.05
        nb, cnt = owner_of(P3)
        if nb is None or nb == part:
            print("%s: сосед не найден (%.1f мм²)" % (part, g.area)); continue
        sec2 = T[nb].section(plane_origin=p - n * 0.02, plane_normal=n)
        if sec2 is not None:
            sh2 = Polygon()
            for P in sorted([Polygon(np.c_[(e - p) @ u, (e - p) @ v]).buffer(0) for e in sec2.discrete if len(e) >= 4], key=lambda P: -P.area):
                sh2 = sh2.symmetric_difference(P)
            g = g.intersection(sh2)
            if isinstance(g, MultiPolygon): g = max(g.geoms, key=lambda q: q.area)
            if g.is_empty or g.area < 0.5: print("%s -> %s: касания нет" % (part, nb)); continue
        rec = {"part": part, "neighbor": nb, "area_mm2": round(g.area, 1)}
        if j["key"] == "none":
            rec["key"] = "клей"; report.append(rec); print("%-12s -> %-12s клей (по плану)" % (part, nb)); continue
        # ориентация — по главным осям среза
        mrr = g.minimum_rotated_rectangle; xy = np.array(mrr.exterior.coords)[:4]
        e1 = xy[1] - xy[0]; e2 = xy[2] - xy[1]
        minor = min(np.linalg.norm(e1), np.linalg.norm(e2)); ang = np.arctan2(e1[1], e1[0])
        from shapely.geometry import Point
        c0 = polylabel(g, 0.01); cands = [(c0.x, c0.y)]
        bx = np.array(g.bounds); st = max(0.3, minor / 12)
        for gx in np.arange(bx[0], bx[2], st):
            for gy in np.arange(bx[1], bx[3], st):
                if g.contains(Point(gx, gy)): cands.append((gx, gy))
        cw = []
        for cx, cy in cands:
            rin = g.boundary.distance(Point(cx, cy))
            cw.append((min(KW * minor, np.sqrt(2) * rin - 2 * (WALL + CS), WMAX), cx, cy))
        cw.sort(key=lambda t: -t[0]); cw = [cw[0]] + [c for c in cw[1:] if c[0] >= WMIN][::max(1, len(cw) // 14)][:14]

        def place(own, rcv):
            """лучший шип из own в rcv: (ok, wk, hk, wt, Mx, ctr, depth)"""
            d = -n if own == part else n                 # направление шипа: в получателя
            ku = np.cos(ang) * u + np.sin(ang) * v; kv = np.cross(d, ku)
            tr = to_tm(M[rcv])                           # получатель с уже вырезанными гнёздами
            depth = 0.0
            for w0, cx, cy in cw:
                ctr = p + cx * u + cy * v
                hit, _, _ = tr.ray.intersects_location([ctr + d * 0.01], [d])
                depth = float(np.min(np.linalg.norm(hit - ctr, axis=1))) if len(hit) else 0.0
                wk = w0
                for _ in range(2):
                    if wk < WMIN: break
                    hk = min(KHj * wk, depth - CT - FLOOR, 0.5 * depth)  # гнездо не глубже половины детали
                    if hk < 0.8: break
                    wt = wk - 2 * hk * tan
                    Mx = mat34(ctr, ku, kv, d)
                    shell = prism(wk + 2 * (CS + WALL), hk + CT + FLOOR, wt + 2 * (CS + WALL), 0.05).transform(Mx)
                    if (shell - M[rcv]).volume() < 0.002 * shell.volume():
                        return True, wk, hk, wt, Mx, ctr, depth, d, ku
                    wk *= 0.85
            return False, 0, 0, 0, None, None, depth, d, ku

        # "auto": гнездо — в деталь, где вокруг стыка больше материала (у профи бывает в обе стороны)
        opts = [(part, nb), (nb, part)] if j["key"] == "auto" else [(part, nb) if j["key"] == "part" else (nb, part)]
        best = None
        for own_, rcv_ in opts:
            r_ = place(own_, rcv_)
            if r_[0] and (best is None or r_[1] * r_[2] > best[1][1] * best[1][2] * 1.15):
                best = ((own_, rcv_), r_)
            if best is None and not r_[0]: depth = r_[6]
        ok = best is not None
        if ok:
            (own, rcv), (_, wk, hk, wt, Mx, ctr, depth, d, ku) = best
        if not ok:
            rec["key"] = "клей (мало места)"; report.append(rec)
            print("%-12s -> %-12s клей: места под шип нет (срез %.1f мм², глубина %.1f)" % (part, nb, g.area, depth)); continue
        ten = prism(wk, hk + 0.15, wt, -0.15).transform(Mx)
        mor = prism(wk + 2 * CS, hk + CT + 0.05, wt + 2 * CS, -0.05).transform(Mx)
        M[own] = M[own] + ten; M[rcv] = M[rcv] - mor
        rec.update(key="шип", owner=own, receiver=rcv, w=round(wk, 2), h=round(hk, 2), top=round(wt, 2),
                   center=np.round(ctr, 2).tolist(), dir=np.round(d, 3).tolist())
        report.append(rec); checks.append((own, rcv, ctr, d, ku, wk, hk))
        print("%-12s -> %-12s шип %.1f×%.1f мм, высота %.1f (срез %.0f мм², глубина гнезда до стенки %.1f)" %
              (own, rcv, wk, wk, hk, g.area, depth))

for nme in T:
    man = M[nme]
    comps = sorted(man.decompose(), key=lambda c: -c.volume())
    if len(comps) > 1 and comps[1].volume() > 0.05:
        print("ВНИМАНИЕ %s: %d тел (%s мм³)" % (nme, len(comps), [round(c.volume(), 2) for c in comps[:4]]))
    M[nme] = comps[0]
    tm = to_tm(M[nme]); tm.export("%s/%s.stl" % (a.outdir, nme))
    print("%-14s %6.0f мм³  %s мм  замкнута %s" % (nme, tm.volume, tm.extents.round(1), tm.is_watertight))
json.dump({"H": a.H, "clear_side": CS, "clear_top": CT, "joints": report}, open(a.outdir + "/keys.json", "w"),
          ensure_ascii=False, indent=1)

# разрезы по каждому шипу
if checks:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = len(checks); nc = min(4, k); nr = (k + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.6 * nr), dpi=110); axs = np.atleast_1d(axs).ravel()
    for ax, (own, rcv, ctr, d, ku, wk, hk) in zip(axs, checks):
        pn = np.cross(d, ku)
        for nm, col in ((own, "#e0604a"), (rcv, "#4a8be0")):
            sc_ = to_tm(M[nm]).section(plane_origin=ctr, plane_normal=pn)
            if sc_ is None: continue
            for e in sc_.discrete:
                x = (e - ctr) @ ku; y = (e - ctr) @ d
                ax.fill(x, y, color=col, alpha=0.55, lw=0); ax.plot(x, y, color=col, lw=0.5)
            ax.plot([], [], color=col, lw=6, label=nm)
        L = max(3 * wk, 2.5 * hk); ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("%s → %s: %.1f мм, h %.1f" % (own, rcv, wk, hk), fontsize=8); ax.legend(fontsize=6)
    for ax in axs[k:]: ax.axis("off")
    plt.tight_layout(); plt.savefig(a.outdir + "/keys_check.png"); plt.close()
from v3lib import render, sheet, PAL
ms = [(n, to_tm(M[n]), PAL[k % len(PAL)]) for k, n in enumerate(J["parts"])]
ims = render(ms, views=("front", "left", "back"), W=420) + render(ms, views=("front", "left", "back"), explode=0.9, W=420)
sheet(ims, 3, legend=[(n, c) for n, _, c in ms], W=420).save(a.outdir + "/preview.png")
print("ok ->", a.outdir)
