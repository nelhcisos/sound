"""Гнёзда под кроксы в торце-подставке + превью сборки с фигуркой.
    python surgeon_sockets.py source/base_original.stl   (trimesh, manifold3d)"""
import sys
from pathlib import Path
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
CAP_T = 7.0
base = trimesh.load(sys.argv[1])

# центр подставки — по нижнему сечению; высота верха — лучами сверху между ног
lo, hi = base.bounds
sec = base.section([0, 0, 1], [0, 0, lo[2] + 0.5])
pts = np.vstack([e for e in sec.discrete])
cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
probe = np.array([[cx + dx, cy + dy, 50.0] for dx, dy in ((0, 12), (0, -12), (12, 8), (-12, 8), (0, 15))])
locs, idx, _ = base.ray.intersects_location(probe, np.tile([0, 0, -1.0], (len(probe), 1)), multiple_hits=False)
tops = locs[:, 2]
h_top = float(np.median(tops))
print(f"центр подставки ({cx:.2f}, {cy:.2f}), верх {h_top:.2f} мм (разброс {tops.min():.2f}..{tops.max():.2f})")

# пустоты гнёзд: цилиндр под верхом минус подставка, оставить куски, выходящие к верху
cyl = trimesh.creation.cylinder(radius=17.0, height=h_top - 0.08 - 1.5, sections=128)
cyl.apply_translation([cx, cy, 1.5 + (h_top - 0.08 - 1.5) / 2])
voids = cyl.difference(base, engine="manifold")
socks = [c for c in voids.split(only_watertight=True) if c.bounds[1][2] > h_top - 0.3 and 1.0 < c.volume < 200]
print("гнёзд:", len(socks), [round(c.volume, 1) for c in socks], [c.bounds.round(2).tolist() for c in socks])
assert len(socks) == 2
sock = trimesh.util.concatenate([s for c in socks for s in (c, c.copy().apply_translation([0, 0, 0.6]))])
sock = trimesh.boolean.union([c for c in sock.split(only_watertight=True)], engine="manifold")

# в систему печати торца: низ (внутренняя сторона) Z=0, верх Z=CAP_T, центр XY=0
to_cap = trimesh.transformations.translation_matrix([-cx, -cy, CAP_T - h_top])
sock.apply_transform(to_cap)
cap = trimesh.load(HERE / "_cap_A_blank.stl")
cap_done = cap.difference(sock, engine="manifold")
print("торец A:", cap_done.is_watertight, round(cap_done.volume / 1000, 2), "мл")
cap_done.export(HERE / "cap_A.stl")

# превью: криптекс вертикально, фигурка на торце
fig = [trimesh.load(f) for f in sorted((HERE / "figure").glob("*.stl"))]
figm = trimesh.util.concatenate(fig)
figm.apply_transform(to_cap)                  # в систему торца (печатной)
M_cap2cx = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)  # печать торца -> криптекс
M_cx2world = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], float)   # ось X криптекса -> вверх
for tag, dx in (("closed", 0.0), ("openA", -70.0)):
    sc = trimesh.load(HERE / f"_scene_{tag}.stl")
    f = figm.copy(); f.apply_transform(M_cap2cx); f.apply_translation([dx, 0, 0])
    capm = cap_done.copy(); capm.apply_transform(M_cap2cx); capm.apply_translation([dx, 0, 0])
    allm = trimesh.util.concatenate([sc, f, capm]); allm.apply_transform(M_cx2world)
    allm.export(HERE / f"_view_{tag}.stl")
    print(tag, "габарит", allm.extents.round(1))
