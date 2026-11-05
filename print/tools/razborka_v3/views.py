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
o.save(a.out); print("ok", a.out, o.size)
