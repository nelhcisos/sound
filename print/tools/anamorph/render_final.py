"""Финальные картинки: вид из точки обзора, ракурсы «распада», схема сборки, раскладка деталей.

python3 render_final.py OUT_FULL OUT_PARTS
"""
import os, sys, glob, json
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import build as B
import scene_moss_line as SC

out_full, out_parts = sys.argv[1], sys.argv[2]
tmp = os.path.join(out_parts, "_r")
os.makedirs(tmp, exist_ok=True)
R = B.compute(verbose=False)
cam = R["cam"]
mc = B.magic_cam(cam, R["stones"])
E, T = np.array(mc["E"]), np.array(mc["T"])

# 1) игровой стиль: точка обзора + распад
S = B.scene_json(R["stones"], R["pads"])
cams = [mc,
        dict(mc, name="shift_5mm", E=(E + [5, 0, 0]).tolist()),
        dict(mc, name="side_right", E=(E + [110, 40, 15]).tolist(), T=[10, 45, 15]),
        dict(mc, name="side_left", E=(E + [-120, 50, 30]).tolist(), T=[-5, 45, 15]),
        dict(mc, name="above", E=[30, -90, 190], T=[0, 42, 8], vfov=46, w=1600, h=1100)]
B.render(S, cams, tmp)

# 2) макет: детали из STL, основание голубое
def stl_mesh(f, color, **kw):
    t = trimesh.load(f)
    return B.mesh_dict(t.vertices, t.faces, color, **kw)

parts = sorted(glob.glob(os.path.join(out_parts, "[0-9][0-9]_*.stl")))
info = {p["num"]: p for p in json.load(open(os.path.join(out_parts, "_info.json")))["parts"]}
meshes = [stl_mesh(os.path.join(out_parts, "00_base.stl"), "#5d9cc4")]
for f in parts:
    n = int(os.path.basename(f)[:2])
    if n == 0:
        continue
    meshes.append(stl_mesh(f, "#d8c7a1"))
# мох поверх (для цвета на макете)
for k, solid in R["pads"].items():
    V, F = B.from_manifold(solid)
    meshes.append(B.mesh_dict(V, F, "#5b9a2e", offset=True))
tv = trimesh.load(os.path.join(out_parts, "viewer_post.stl"))
Sm = dict(meshes=meshes, bg="#f2f0ea", hemi=1.5)
top = dict(name="map", E=[0, 40, 400], T=[0, 40.001, 0], ortho=56, w=1800, h=1000)
Bm = dict(meshes=meshes + [B.mesh_dict(tv.vertices, tv.faces, "#e0b040")], bg="#f2f0ea")
B.render(Sm, [top, dict(mc, name="model_magic")], tmp)
B.render(Bm, [dict(name="model_whole", E=[160, -260, 170], T=[0, -40, 10], vfov=34, w=1600, h=1000)], tmp)

# 3) схема сборки с номерами (ортографический вид сверху: 1 мм = px)
im = Image.open(os.path.join(tmp, "map.png")).convert("RGB")
W, H = im.size
scale = H / (2 * 56)
d = ImageDraw.Draw(im)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except OSError:
    font = fsm = ImageFont.load_default()
import print_parts as PP
pieces = PP.make_pieces(R)
for pc in pieces:
    pts = np.vstack([R["vis"][i].geometry()[0] for i in pc["stones"]])
    top_pts = pts[pts[:, 2] > pts[:, 2].max() - 3]
    x, y = pts[:, 0].mean(), pts[:, 1].mean()
    px, py = W / 2 + x * scale, H / 2 - (y - 40) * scale
    txt = str(pc["num"])
    bb = d.textbbox((0, 0), txt, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    col = (40, 110, 30) if pc["moss"] else (60, 60, 60)
    d.ellipse([px - 19, py - 19, px + 19, py + 19], fill=(255, 255, 255), outline=col, width=3)
    d.text((px - w / 2 - bb[0], py - h / 2 - bb[1]), txt, fill=col, font=font)
d.text((20, H - 40), "Схема сборки, вид сверху. Зелёные номера — камни с мхом. Глаз — внизу, по стрелке-вырезу.", fill=(40, 40, 40), font=fsm)
im.save(os.path.join(out_parts, "preview.png"))

# 4) сводная картинка для папки целой модели
im = Image.open(os.path.join(tmp, "magic.png")).convert("RGB")
im.save(os.path.join(out_full, "preview.png"))
views = os.path.join(out_full, "views")
os.makedirs(views, exist_ok=True)
for n in ("magic", "shift_5mm", "side_right", "side_left", "above", "model_magic", "model_whole"):
    Image.open(os.path.join(tmp, n + ".png")).convert("RGB").save(os.path.join(views, n + ".png"))
for f in glob.glob(os.path.join(tmp, "*")):
    os.remove(f)
os.rmdir(tmp)
print("ok")
