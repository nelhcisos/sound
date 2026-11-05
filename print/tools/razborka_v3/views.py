"""Осмотр мастера для выбора плоскостей: 4 вида с сеткой (мм) + горизонтальные/вертикальные сечения.

usage: python3 views.py master.stl out.png [--grid 5] [--z 10,20,30] [--x 0] [--y 0]
Сечения рисуются контурами с сеткой 5 мм: по ним видно, где ноги/руки отдельны, где подол, толщины.
"""
import sys, os, argparse
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3lib import render, sheet

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--grid", type=float, default=5); ap.add_argument("--z", default=""); ap.add_argument("--x", default="")
ap.add_argument("--y", default=""); ap.add_argument("--W", type=int, default=520)
ap.add_argument("--profile", default="", help="профиль площади сечения по z; 'all' или область x0,x1,y0,y1")
a = ap.parse_args()
m = trimesh.load(a.src, force="mesh")
ims = render([("m", m, (200, 200, 210))], W=a.W, grid=a.grid)
from PIL import Image
o = sheet(ims, 4, W=a.W)
cuts = [("z", float(v)) for v in a.z.split(",") if v] + [("x", float(v)) for v in a.x.split(",") if v] + [("y", float(v)) for v in a.y.split(",") if v]
if cuts:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nc = min(4, len(cuts)); nr = (len(cuts) + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), dpi=100); axs = np.atleast_1d(axs).ravel()
    for ax, (k, v) in zip(axs, cuts):
        i = "xyz".index(k); n = np.eye(3)[i]; s = m.section(plane_origin=n * v, plane_normal=n)
        a2 = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[i]
        if s is not None:
            for e in s.discrete:
                ax.fill(e[:, a2[0]], e[:, a2[1]], color="#e07050", alpha=0.35); ax.plot(e[:, a2[0]], e[:, a2[1]], "k", lw=0.6)
        ax.set_title("%s = %g" % (k, v)); ax.set_aspect("equal"); ax.grid(alpha=0.4)
        ax.set_xlabel("xyz"[a2[0]]); ax.set_ylabel("xyz"[a2[1]])
        ax.xaxis.set_major_locator(plt.MultipleLocator(a.grid)); ax.yaxis.set_major_locator(plt.MultipleLocator(a.grid))
        ax.tick_params(labelsize=6)
    for ax in axs[len(cuts):]: ax.axis("off")
    plt.tight_layout(); tmp = a.out + ".sec.png"; plt.savefig(tmp); plt.close()
    sec = Image.open(tmp).convert("RGB"); os.remove(tmp)
    sec.thumbnail((o.width, 10 ** 5))
    o2 = Image.new("RGB", (o.width, o.height + sec.height), "white"); o2.paste(o, (0, 0)); o2.paste(sec, (0, o.height)); o = o2
if a.profile:
    # площадь сечения по высоте: скачок = край (подол, манжета, верх обуви), провал = перетяжка (шея, запястье)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, box
    reg = None
    if a.profile != "all":
        x0, x1, y0, y1 = [float(t) for t in a.profile.split(",")]
        reg = box(x0, y0, x1, y1)
    zs = np.arange(m.bounds[0][2] + 0.2, m.bounds[1][2] - 0.2, 0.25); A = []
    for z in zs:
        sc_ = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1]); ar = 0.0
        if sc_ is not None:
            sh = Polygon()
            for P in sorted([Polygon(e[:, :2]).buffer(0) for e in sc_.discrete if len(e) >= 4], key=lambda P: -P.area):
                sh = sh.symmetric_difference(P)
            ar = (sh.intersection(reg) if reg is not None else sh).area
        A.append(ar)
    A = np.array(A); dA = np.gradient(np.log(A + 1.0), zs)   # относительный скачок, 1/мм
    from scipy.signal import find_peaks
    pk, _ = find_peaks(np.abs(dA), height=0.04, prominence=0.03, distance=6)
    mn, _ = find_peaks(-A, prominence=A.max() * 0.02, distance=6)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), dpi=100)
    ax[0].plot(zs, A, "k"); ax[0].set_title("площадь сечения, мм² (%s); зелёные — перетяжки" % a.profile); ax[0].grid(alpha=0.4)
    ax[1].plot(zs, dA, "b"); ax[1].set_title("относительный скачок площади, 1/мм: красные — края (подол, манжета, обувь)")
    ax[1].grid(alpha=0.4); ax[1].set_ylim(-0.6, 0.6)
    for i in pk:
        for axx in ax: axx.axvline(zs[i], color="r", lw=0.6)
        ax[1].text(zs[i], np.clip(dA[i], -0.55, 0.55), " %.1f" % zs[i], color="r", fontsize=7)
    for i in mn:
        ax[0].axvline(zs[i], color="g", lw=0.8); ax[0].text(zs[i], A[i], " %.1f" % zs[i], color="g", fontsize=7)
    for axx in ax: axx.xaxis.set_major_locator(plt.MultipleLocator(5)); axx.tick_params(labelsize=7)
    plt.tight_layout(); tmp = a.out + ".prof.png"; plt.savefig(tmp); plt.close()
    pr = Image.open(tmp).convert("RGB"); os.remove(tmp); pr.thumbnail((o.width, 10 ** 5))
    o2 = Image.new("RGB", (o.width, o.height + pr.height), "white"); o2.paste(o, (0, 0)); o2.paste(pr, (0, o.height)); o = o2
    print("края (скачки площади) на z:", [round(float(zs[i]), 1) for i in pk], " перетяжки на z:", [round(float(zs[i]), 1) for i in mn])
o.save(a.out); print("ok", a.out, o.size)
