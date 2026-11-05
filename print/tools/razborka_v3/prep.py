"""Подготовка мастера для разборки v3: сплошная замкнутая сетка 100 мм, Z вверх, лицо -> -Y, дно на z=0.

usage: python3 prep.py in.(glb|stl|obj) master.stl [опции]
  --up y|z        ось «вверх» у исходника (GLB обычно y; по умолчанию: glb -> y, остальное -> z)
  --rotz DEG      повернуть вокруг Z, чтобы лицо смотрело в -Y (проверять по views.py)
  --hollow        вход — наша полая STL со сливом (hollow.py): заткнуть слив и выбросить полость
  --voxel PITCH   принудительно пересобрать через воксели (мм на мастере, напр. 0.12) — для «грязных» сеток
  --height H      высота мастера (по умолчанию 100)
"""
import sys, os, argparse
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3lib import to_man, to_tm
import manifold3d as m3

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--up", default=None); ap.add_argument("--rotz", type=float, default=0)
ap.add_argument("--hollow", action="store_true"); ap.add_argument("--voxel", type=float, default=0)
ap.add_argument("--height", type=float, default=100.0)
a = ap.parse_args()

s = trimesh.load(a.src, force="scene")
m = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
m = trimesh.Trimesh(m.vertices, m.faces, process=True)
up = a.up or ("y" if a.src.lower().endswith((".glb", ".gltf")) else "z")
if up == "y":
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
if a.rotz:
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(a.rotz), [0, 0, 1]))
print("исходник: граней %d, размер %s, замкнута %s" % (len(m.faces), m.extents.round(2), m.is_watertight))


def voxel_remesh(m, pitch):
    from scipy import ndimage
    from skimage import measure
    vox = m.voxelized(pitch); g = vox.matrix
    D = 3; g = np.pad(g, D + 2)
    filled = ndimage.binary_fill_holes(ndimage.binary_dilation(g, iterations=D))
    solid = ndimage.binary_erosion(filled, iterations=D) | (g & filled)
    f = ndimage.gaussian_filter(np.pad(solid, 1).astype(np.float32), 0.7)
    v, fc, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 1 - (D + 2)) * pitch + vox.translation
    out = trimesh.Trimesh(v, fc, process=True)
    if out.volume < 0: out.invert()
    return out


if a.hollow:
    # слив hollow.py: вертикальный цилиндр в дне. Ищем маленькую дыру в сечении у дна.
    z0 = m.bounds[0][2]
    sec = m.section(plane_origin=[0, 0, z0 + 0.25], plane_normal=[0, 0, 1])
    pl, T = sec.to_planar()
    holes = []
    for poly in pl.polygons_full:
        for ring in poly.interiors:
            from shapely.geometry import Polygon
            h = Polygon(ring)
            if h.area < 30: holes.append(h)
    if not holes: raise SystemExit("слив не найден")
    man = to_man(m)
    for h in holes:
        c2 = np.array(h.centroid.coords[0]); r = np.sqrt(h.area / np.pi) + 0.3
        c3 = (T @ np.r_[c2, 0, 1])[:3]
        # высота канала: поднимаемся, пока в сечении есть маленькая дыра рядом с центром слива
        ztop = z0 + 0.25
        for zz in np.arange(z0 + 0.25, z0 + 10, 0.1):
            sz = m.section(plane_origin=[0, 0, zz], plane_normal=[0, 0, 1])
            found = False
            if sz is not None:
                plz, Tz = sz.to_planar()
                for poly in plz.polygons_full:
                    for ring in poly.interiors:
                        hh = Polygon(ring); cc = (Tz @ np.r_[hh.centroid.coords[0], 0, 1])[:3]
                        if hh.area < 30 and np.hypot(cc[0] - c3[0], cc[1] - c3[1]) < r: found = True
            if not found: break
            ztop = zz
        plug = m3.Manifold.cylinder(ztop + 0.15 - (z0 - 1.0), r, r, 64).translate([c3[0], c3[1], z0 - 1.0])
        print("канал слива до z=%.2f" % ztop)
        man = man + plug
        print("слив: центр (%.2f, %.2f), Ø%.2f мм — заткнут" % (c3[0], c3[1], 2 * (r - 0.3)))
    man = man.trim_by_plane([0, 0, 1], float(z0))  # затычка не ниже дна
    comps = sorted(man.decompose(), key=lambda c: -c.volume())
    print("оболочек после затыкания:", len(comps), "объёмы:", [round(c.volume(), 1) for c in comps[:4]])
    man = comps[0]
    # полость: внутренняя поверхность входит в тот же Manifold как дыра — проверяем по объёму
    tm = to_tm(man)
    parts = tm.split(only_watertight=False)
    if len(parts) > 1:
        parts = sorted(parts, key=lambda p: -abs(p.volume))
        tm = parts[0]
        print("выброшена внутренняя поверхность полости (%d оболочек -> 1)" % len(parts))
    m = tm

sc = a.height / m.extents[2]
m.apply_scale(sc)
m.apply_translation([-(m.bounds[0][0] + m.bounds[1][0]) / 2, -(m.bounds[0][1] + m.bounds[1][1]) / 2, -m.bounds[0][2]])

ok = False
if not a.voxel:
    try:
        man = to_man(m); ok = True
    except RuntimeError as e:
        print("не manifold (%s) — пересобираю через воксели 0.12 мм" % e)
if not ok:
    m = voxel_remesh(m, a.voxel or 0.12)
    man = to_man(m)
comps = sorted(man.decompose(), key=lambda c: -c.volume())
if len(comps) > 1:
    print("кусков: %d, оставляю крупнейший (остальные объёмом %s мм³)" % (len(comps), [round(c.volume(), 2) for c in comps[1:6]]))
man = comps[0]
out = to_tm(man)
out.export(a.dst)
print("мастер: %s, граней %d, объём %.0f мм³, замкнута %s" % (out.extents.round(2), len(out.faces), out.volume, out.is_watertight))
