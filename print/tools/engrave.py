"""Гравировка надписи-картинки на дне модели.
usage: engrave.py in.stl logo.png out.stl [ширина_мм=20] [глубина=0.4] [срез_дна=0.35] [центр_y=1.5]"""
import sys, numpy as np, trimesh, manifold3d as mf
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon
from shapely.ops import unary_union
src, logo, dst = sys.argv[1:4]
W = float(sys.argv[4]) if len(sys.argv) > 4 else 20
DEPTH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
CUT = float(sys.argv[6]) if len(sys.argv) > 6 else 0.35
CY = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5

img = np.asarray(Image.open(logo).convert("L"), float) / 255
ink = img < 0.5
rs, cs = np.nonzero(ink)
img = img[rs.min()-3:rs.max()+4, cs.min()-3:cs.max()+4]
s = W / img.shape[1]
geo = None
for c in measure.find_contours(img, 0.5):
    if len(c) < 4: continue
    p = Polygon(np.c_[c[:, 1] * s, c[:, 0] * s]).buffer(0)  # x=col, y=row (row вниз -> +Y, верх надписи к лицу фигурки)
    geo = p if geo is None else geo.symmetric_difference(p)
geo = geo.buffer(0.08).buffer(-0.02).simplify(0.02)  # тонкие волосяные линии чуть толще
b = geo.bounds
from shapely import affinity
geo = affinity.translate(geo, -(b[0]+b[2])/2, -(b[1]+b[3])/2 + CY)
print("надпись мм:", np.round(np.array(geo.bounds), 2))

m = trimesh.load(src)
man = lambda t: mf.Manifold(mf.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
M = man(m)
# ровное дно
box = trimesh.creation.box(bounds=[[-50, -50, CUT], [50, 50, 100]])
M = M ^ man(box)
# вырез надписи
polys = list(geo.geoms) if hasattr(geo, "geoms") else [geo]
cut = None
for p in polys:
    if p.area < 0.005: continue
    t = trimesh.creation.extrude_polygon(p, CUT + DEPTH + 1)
    t.apply_translation([0, 0, -1])
    k = man(t); cut = k if cut is None else cut + k
M = M - cut
o = M.to_mesh()
out = trimesh.Trimesh(o.vert_properties[:, :3], o.tri_verts)
out.apply_translation([0, 0, -out.bounds[0][2]])
print("размер мм:", out.extents.round(2), "замкнута:", out.is_watertight, "граней:", len(out.faces))
out.export(dst)
