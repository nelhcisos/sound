"""Проверка печати на смоле без поддержек: режем деталь на слои (как слайсер) и ищем
острова (кусок слоя, не опирающийся на предыдущий) и широкие уступы-нависания.

    python check_print.py <папка со STL> [толщина слоя, мм]
Нужны: trimesh, shapely, numpy (pip install trimesh shapely numpy rtree scipy networkx).
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

folder = Path(sys.argv[1])
LH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05


def layer_polys(mesh, zs):
    secs = mesh.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=zs)
    out = []
    for s in secs:
        if s is None:
            out.append(Polygon())
            continue
        polys = [p.buffer(0) for p in s.polygons_full]
        out.append(unary_union(polys) if polys else Polygon())
    return out


def parts(geom):
    if geom.is_empty:
        return []
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


for stl in sorted(folder.glob("*.stl")):
    m = trimesh.load(stl)
    bodies = m.split(only_watertight=False)
    z0, z1 = m.bounds[0][2], m.bounds[1][2]
    zs = np.arange(z0 + LH / 2, z1, LH)
    layers = layer_polys(m, zs)
    first = layers[0].area
    islands, ledges = [], []
    for k in range(1, len(layers)):
        prev, cur = layers[k - 1], layers[k]
        for p in parts(cur):
            if p.area < 0.01:
                continue
            if not p.intersects(prev):
                islands.append((round(zs[k], 2), round(p.area, 2)))
        new = cur.difference(prev.buffer(LH * 1.2))   # прирост больше, чем у скоса 45°
        for p in parts(new):
            if p.area < 0.05:
                continue
            # вылет уступа: самая дальняя точка нового куска от опоры (предыдущего слоя)
            pts = list(p.exterior.coords)
            reach = max(prev.distance(Point(c)) for c in pts)
            ledges.append((round(zs[k], 2), round(p.area, 2), round(reach, 2)))
    big = [l for l in ledges if l[2] > 0.3]
    worst = max((l[2] for l in ledges), default=0.0)
    print(f"{stl.name:20s} замкнута={m.is_watertight} тел={len(bodies)} объём={m.volume/1000:.1f} мл "
          f"высота={z1-z0:.1f} мм  площадь на столе={first:.0f} мм²  островов={len(islands)}  "
          f"макс. вылет уступа={worst:.2f} мм")
    for z, a in islands[:5]:
        print(f"    ОСТРОВ на z={z}: {a} мм²")
    seen = set()
    for z, a, r in sorted(big, key=lambda t: -t[2]):
        key = round(r, 1)
        if key in seen:
            continue
        seen.add(key)
        print(f"    уступ z={z} мм: площадь {a} мм², вылет {r} мм")
        if len(seen) >= 4:
            break
