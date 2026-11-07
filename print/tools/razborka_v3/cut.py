"""Шаг 2 разборки v3: плоские срезы на мастере 100 мм.

usage: python3 cut.py plan.json outdir

plan.json:
{
  "master": "путь/master.stl",
  "rest": "jacket",                       # имя того, что останется после всех срезов
  "parts": [                              # порядок = порядок снятия (от краёв к центру)
    {"name": "base",
     "planes": [{"p": [0, 0, 7.2], "n": "-z"}],   # 1-я плоскость = плоскость стыка (на ней шип);
                                                  # n смотрит ВНУТРЬ снимаемой детали; "x","-y",... или [x,y,z]
     "cyl": {"c": [x, y, z], "r": 9, "axis": "z"},# необязательно: локальный резак — цилиндр (ось по умолчанию = n)
     "box": [[x0,y0,z0], [x1,y1,z1]],             # необязательно: ограничить коробкой
     "prism": {"at": [x,y,z], "axis": "z", "margin": 0.4, "pick": [x,y,z]},  # сечение тела +margin, вытянутое по оси
     # вместо planes/...: "any": [{область}, {область}] — объединение областей (плоскость стыка = 1-я плоскость 1-й области)
     # "ellipsoid": {"c": [...], "r": [...], "outside": true, "offset": 0.05} — снаружи черепа (шапка, волосы, маска)
     # "minus": [{область}, ...] — вычесть области (нога, уходящая внутрь ботинка, остаётся у штанов)
     "seed": [0, 0, 3],                           # точка внутри детали: берётся только её кусок
     "key": "auto"}                               # чей шип: "auto" (по умолчанию) — гнездо туда, где толще;
  ]                                               # "part" — из этой детали в соседа, "body" — из соседа в неё, "none" — клей
}
Результат: outdir/master_100mm/*.stl, outdir/joints.json, outdir/scheme.png (в сборе / в разборе).
"""
import sys, os, json
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3lib import to_man, to_tm, cutter, vec, render, sheet, PAL, joint_plane
from scipy.spatial import cKDTree

plan = json.load(open(sys.argv[1])); out = sys.argv[2]
os.makedirs(out + "/master_100mm", exist_ok=True)
base_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
mpath = plan["master"] if os.path.isabs(plan["master"]) else os.path.join(base_dir, plan["master"])
tm = trimesh.load(mpath, force="mesh")
body = to_man(tm); lo, hi = tm.bounds
print("мастер: объём %.0f мм³" % body.volume())

parts = []; joints = []; multi = []
for spec in plan["parts"]:
    name = spec["name"]
    C = cutter(spec, lo, hi, to_tm(body) if ('prism' in json.dumps(spec)) else None)
    A = body ^ C
    comps = [c for c in A.decompose() if c.volume() > 1e-3]
    if not comps:
        raise SystemExit("%s: резак ничего не задел — проверь плоскость" % name)
    if spec.get("seed") == "all":           # деталь из нескольких кусков (пряди чёлки) — берём все
        part = comps[0]
        for c_ in comps[1:]: part = part + c_
        comps = [part]
    if "seed" in spec and spec["seed"] != "all" and len(comps) > 1:
        s = np.asarray(spec["seed"], float)
        def dist(c):
            t = to_tm(c)
            if t.contains([s])[0]: return -1.0
            return cKDTree(t.vertices).query(s)[0]
        ds = [dist(c) for c in comps]; i = int(np.argmin(ds))
    else:
        i = int(np.argmax([c.volume() for c in comps]))
    part = comps[i]; others = [c for j, c in enumerate(comps) if j != i]
    body = body - C
    for o in others:  # чужие куски из резака возвращаем телу
        body = body + o
    big = [o for o in others if o.volume() > 0.5]
    print("%-14s %8.0f мм³" % (name, part.volume()) + ("   в резак попало чужое — возвращено телу: " + ", ".join(
        "%.0f мм³ у %s" % (o.volume(), np.round(to_tm(o).centroid, 1).tolist()) for o in big) if big else ""))
    parts.append((name, part))
    pl = joint_plane(spec)
    if spec.get("seed") == "all": multi.append(name)
    joints.append({"part": name, "p": list(map(float, pl["p"])), "n": list(map(float, vec(pl["n"]))),
                   "key": spec.get("key", "auto"), **({"kh": spec["kh"]} if "kh" in spec else {})})

rest = plan.get("rest", "body")
bc = sorted(body.decompose(), key=lambda c: -c.volume())
for c in bc[1:]:
    if c.volume() > 0.05:
        print("ВНИМАНИЕ: оторванный кусок %.1f мм³ у %s — проверь плоскость рядом (кусок выброшен)" % (c.volume(), np.round(to_tm(c).centroid, 1).tolist()))
parts.append((rest, bc[0]))
print("%-14s %8.0f мм³" % (rest, bc[0].volume()))

meshes = []
for k, (n, man) in enumerate(parts):
    t = to_tm(man); t.export("%s/master_100mm/%s.stl" % (out, n)); meshes.append((n, t, PAL[k % len(PAL)]))
json.dump({"master_height": float(hi[2] - lo[2]), "joints": joints, "parts": [n for n, _ in parts], "multi": multi},
          open(out + "/joints.json", "w"), ensure_ascii=False, indent=1)

ims = render(meshes, W=380) + render(meshes, views=("front", "left", "back", "right"), explode=0.9, W=380)
sheet(ims, 4, legend=[(n, c) for n, _, c in meshes], W=380).save(out + "/scheme.png")
print("ok ->", out)
