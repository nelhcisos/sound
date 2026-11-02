"""Цельная одноцветная версия: подложка + все камни одним куском, мох — углублениями.

python3 print_mono.py OUTDIR [длина_мм=80]
Глубина канавки, толщина подложки, стенки — в мм итоговой модели.
"""
import os, sys, json
import numpy as np
import manifold3d as mf
from shapely.geometry import Point
from shapely.ops import unary_union
from anamorph import convex_manifold, front_facets, moss_recess, Camera
import build as B
import print_parts as PP
import scene_moss_line as SC

OUT = sys.argv[1]
LEN = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
DEPTH = float(os.environ.get("DEPTH", 0.8))  # глубина углубления мха, мм итог
TEXTURE = os.environ.get("TEXTURE", "none")  # none — гладкое углубление, dimples — лунки на дне
HATCH = 0.8     # шаг лунок, мм итог
HATCH_D = 0.5   # глубина лунок сверх канавки
DIMPLE_D = 0.5  # диаметр лунки, мм итог
BASE = 3.0      # толщина подложки, мм итог
WALL = 1.2      # стенка полостей, мм итог
DRAIN = 1.5
os.makedirs(OUT, exist_ok=True)

R = B.compute()
cam = R["cam"]
pieces = PP.make_pieces(R)
for pc in pieces:
    pc["footprint"] = unary_union([PP.section(R["vis"][i], 0.0) for i in pc["stones"]])
outline = PP.base_outline(pieces)
s = LEN / (outline.bounds[2] - outline.bounds[0])  # масштаб модель → итог
u = 1 / s                                          # 1 мм итога в единицах модели

plate = PP.prism(outline, -BASE * u, -0.8 * u) + PP.prism(outline.buffer(-0.8 * u), -0.8 * u, 0.0)
rocks = mf.Manifold.batch_boolean([convex_manifold(st.clip_z(-1.0).grow(0.09)) for st in R["vis"]], mf.OpType.Add)
# фактура дна мха
from shapely.geometry import box as _box
from shapely import affinity as _aff
dep_ref = float(np.linalg.norm(SC.LOOK - cam.E))
per = HATCH * u / dep_ref
bx = R["pat"].bounds
cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
L = max(bx[2] - bx[0], bx[3] - bx[1]) * 1.5
# вдавления: сотовая сетка круглых лунок в плоскости изображения
_pts = []
_r = DIMPLE_D / 2 * u / dep_ref
for j in range(int(2 * L / (per * 0.866)) + 1):
    yy = cy - L + j * per * 0.866
    off = per / 2 if j % 2 else 0.0
    for i in range(int(2 * L / per) + 1):
        _pts.append(Point(cx - L + off + i * per, yy))
_pts = [p for p in _pts if R["pat"].distance(p) < _r * 2]
stripes = unary_union([p.buffer(_r, quad_segs=6) for p in _pts])
grooves = []
for k, M in enumerate(R["regions"]):
    if M is None or k not in R["pads"]:
        continue
    st = R["vis"][k]
    ff = front_facets(cam, st)
    g = moss_recess(cam, st, M, ff, depth=DEPTH * u, lift=0.3 * u)
    h = moss_recess(cam, st, M.intersection(stripes), ff, depth=(DEPTH + HATCH_D) * u, lift=0.3 * u) if TEXTURE == "dimples" else None
    for x in (g, h):
        if x is not None:
            grooves.append(x)
groove = mf.Manifold.batch_boolean(grooves, mf.OpType.Add)
body = plate + rocks - groove

# полости в крупных камнях + сливы насквозь через дно подложки
nh = 0
for st in R["vis"]:
    inner = st.grow(-WALL * u).clip_z(0.6 * u)
    if inner.interior()[1] < 0.5 * u or inner.volume() * s ** 3 < 60:
        continue
    zb = float(inner.geometry()[0][:, 2].min())
    sec = PP.section(inner, zb + 0.02 * u).buffer(-(DRAIN / 2 + 0.3) * u)
    if sec.is_empty or sec.area < (1.0 * u) ** 2:
        continue
    cav = convex_manifold(inner)
    for h in PP.far_pair(sec):
        cav = cav + mf.Manifold.cylinder(zb + (BASE + 1) * u, DRAIN / 2 * u, DRAIN / 2 * u, 24).translate((h[0], h[1], -(BASE + 0.5) * u))
        nh += 1
    body = body - cav

# подпись на дне: где глаз (мм итога, от стола)
E = cam.E * s
b = outline.bounds
for kline, line in enumerate([f"ГЛАЗ x{E[0]:+.0f} y{E[1] - b[1] * s:+.0f} h{E[2] + BASE:.0f}", "мм от перед. края"]):
    cs, _ = PP.text_cs(line, 3.0 * u, np.array([(b[0] + b[2]) / 2, b[1] + (12 - 5 * kline) * u]), mirror=True)
    body = body - mf.Manifold.extrude(cs, 0.5 * u).translate((0, 0, -(BASE + 0.1) * u))

final = body.scale((s, s, s))
comps = sorted(final.decompose(), key=lambda m: -m.volume())
for c in comps[1:]:
    print("отдельный кусок:", round(c.volume(), 3), "мм³", [round(x, 2) for x in c.bounding_box()])
final = comps[0]
import trimesh
_m = final.to_mesh64()
_t = trimesh.Trimesh(np.asarray(_m.vert_properties)[:, :3], np.asarray(_m.tri_verts), process=False)
_t.export(os.path.join(OUT, f"moss_line_rocks_mono_{LEN:.0f}mm.3mf"))
print("3mf topology watertight:", _t.is_watertight)
t = PP.save(final, os.path.join(OUT, f"moss_line_rocks_mono_{LEN:.0f}mm.stl"))

# глазок в итоговом размере
cam_s = Camera(cam.E * s, SC.LOOK * s)
PP.BASE_T = BASE
v = PP.save(PP.build_viewer(cam_s), os.path.join(OUT, f"viewer_post_{LEN:.0f}mm.stl"))
info = dict(scale=round(s, 4), size=[round(float(e), 1) for e in t.extents], volume_ml=round(t.volume / 1000, 2),
            watertight=bool(t.is_watertight), drains=nh, groove_depth=DEPTH, base=BASE,
            eye_from_front_edge_mm=round(float(-(E[1] - b[1] * s)), 1), eye_x=round(float(E[0]), 1),
            eye_h_table=round(float(E[2] + BASE), 1), viewer_watertight=bool(v.is_watertight))
json.dump(info, open(os.path.join(OUT, "_mono_info.json"), "w"), indent=1)
print(info)
