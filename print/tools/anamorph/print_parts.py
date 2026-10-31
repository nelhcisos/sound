"""Детали для печати: камни со штырём-«подошвой», основание с гнёздами, глазок, цельная версия.

python3 print_parts.py OUT_FULL OUT_PARTS
Все размеры — мм итоговой модели (смола: зазор 0.15).
"""
import os, sys, json
import numpy as np
import manifold3d as mf
import trimesh
from shapely.geometry import MultiPoint, Polygon, Point, box
from shapely.ops import unary_union, polylabel
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

from anamorph import convex_manifold, moss_solid, front_facets, from_manifold, unit
import scene_moss_line as SC
import build as B

CLEAR = float(os.environ.get("CLEAR", 0.15))   # радиальный зазор гнезда
PEG_H = 1.5        # высота «подошвы» в гнезде
BASE_T = 4.0       # толщина основания
WALL = 1.4         # стенка полых камней
FLOOR_Z = 1.0      # пол полости (над водой)
DRAIN_D = 1.5      # слив смолы
ENGR = 0.4         # глубина гравировки
HOLLOW_MIN = 250.0  # мм³ — меньше не полим


def cs_from_shapely(g):
    rings = []
    for p in (g.geoms if hasattr(g, "geoms") else [g]):
        if p.is_empty:
            continue
        rings.append(np.asarray(p.exterior.coords)[:-1])
        rings += [np.asarray(r.coords)[:-1] for r in p.interiors]
    return mf.CrossSection(rings, mf.FillRule.EvenOdd)


def prism(g, z0, z1):
    return mf.Manifold.extrude(cs_from_shapely(g), z1 - z0).translate((0, 0, z0))


def text_cs(s, height, center, mirror=False, angle=0.0):
    tp = TextPath((0, 0), s, size=10, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    polys = [np.asarray(p, float) for p in tp.to_polygons() if len(p) >= 3]
    allp = np.vstack(polys)
    mn, mx = allp.min(0), allp.max(0)
    k = height / (mx[1] - mn[1])
    c = (mn + mx) / 2
    ca, sa = np.cos(angle), np.sin(angle)
    Rm = np.array([[ca, -sa], [sa, ca]])
    out = []
    for p in polys:
        q = (p - c) * k
        if mirror:
            q[:, 0] *= -1
        out.append(q @ Rm.T + center)
    return mf.CrossSection(out, mf.FillRule.EvenOdd), (mx - mn) * k


def section(stone, z):
    V, _ = stone.add([0, 0, -1, -(z - 1e-4)]).add([0, 0, 1, z + 1e-4]).geometry()
    return MultiPoint(V[:, :2]).convex_hull if len(V) > 2 else Polygon()


def to_tm(m):
    m = m.set_tolerance(0.002).simplify(0.002)
    V, F = from_manifold(m)
    t = trimesh.Trimesh(V, F, process=True)
    if not t.is_watertight:  # добить мелкие щели от округления float32
        t.merge_vertices(digits_vertex=4)
        t.update_faces(t.nondegenerate_faces())
        t.fill_holes()
        t.fix_normals()
    return t


def save(m, path):
    t = to_tm(m)
    t.export(path)
    return t


def far_pair(poly, frac=0.38):
    c = np.array(poly.centroid.coords[0])
    P = np.asarray(poly.exterior.coords)[:-1] - c
    w, v = np.linalg.eigh(P.T @ P)
    ax = v[:, -1]
    ext = np.abs(P @ ax).max()
    pts = []
    for sgn in (-1, 1):
        q = Point(c + sgn * frac * ext * ax)
        if not poly.contains(q):
            from shapely.ops import nearest_points
            q = nearest_points(poly, q)[0]
        pts.append(np.array(q.coords[0]))
    return pts


def make_pieces(R):
    stones, vis, pads = R["stones"], R["vis"], R["pads"]
    groups = {}
    for i, s in enumerate(stones):
        if s.meta.get("kind") == "cliff":
            groups.setdefault(("cliff", s.meta["block"]), []).append(i)
        else:
            groups[(s.meta.get("kind"), i)] = [i]

    def cx(idx):
        return float(np.mean([vis[i].geometry()[0][:, 0].mean() for i in idx]))

    def cy(idx):
        return float(np.mean([vis[i].geometry()[0][:, 1].mean() for i in idx]))

    cliff = sorted([g for g in groups if g[0] == "cliff"], key=lambda g: cx(groups[g]))
    mid = sorted([g for g in groups if g[0] in ("key", "mid")], key=lambda g: cx(groups[g]))
    front = sorted([g for g in groups if g[0] == "front"], key=lambda g: cx(groups[g]))
    pieces = []
    for kind, lst in (("cliff", cliff), ("rock", mid), ("pebble", front)):
        for g in lst:
            pieces.append(dict(kind=kind, stones=groups[g], num=len(pieces) + 1,
                               moss=any(i in pads for i in groups[g])))
    return pieces


def build_piece(R, pc):
    vis, pads = R["vis"], R["pads"]
    idx = pc["stones"]
    grow = 0.01 if len(idx) > 1 else 0.0
    above = mf.Manifold.batch_boolean([convex_manifold(vis[i].grow(grow)) for i in idx], mf.OpType.Add)
    fp = unary_union([section(vis[i], 0.0) for i in idx])
    fp = fp.buffer(0.005).buffer(-0.005)
    body = above + prism(fp.buffer(-0.05, join_style="mitre"), -PEG_H, 0.3)
    moss = [pads[i] for i in idx if i in pads]
    if moss:
        body = body + mf.Manifold.batch_boolean(moss, mf.OpType.Add)
    holes_xy = []
    cav_vol = 0.0
    for i in idx:
        inner = vis[i].grow(-WALL).clip_z(FLOOR_Z)
        if inner.interior()[1] < 1.0:
            continue
        v = inner.volume()
        if v < HOLLOW_MIN:
            continue
        cav = convex_manifold(inner)
        zb = float(inner.geometry()[0][:, 2].min())
        sec = section(inner, zb + 0.05).buffer(-(DRAIN_D / 2 + 0.25))
        if sec.is_empty or sec.area < 1.0:
            continue
        hs = far_pair(sec)
        for h in hs:
            cav = cav + mf.Manifold.cylinder(zb + PEG_H + 0.8, DRAIN_D / 2, DRAIN_D / 2, 24).translate((h[0], h[1], -PEG_H - 0.2))
            holes_xy.append(h)
        body = body - cav
        cav_vol += v
    # номер на дне (читается снизу)
    avoid = unary_union([Point(h).buffer(DRAIN_D / 2 + 0.9) for h in holes_xy]) if holes_xy else Polygon()
    free = fp.buffer(-0.9).difference(avoid)
    label = None
    if not free.is_empty and free.area > 4:
        g = max(free.geoms, key=lambda p: p.area) if hasattr(free, "geoms") else free
        c = np.array(polylabel(g, 0.05).coords[0])
        r = g.exterior.distance(Point(c))
        for hgt in (5.0, 4.0, 3.2, 2.6, 2.0):
            cs, ext = text_cs(str(pc["num"]), hgt, c, mirror=True)
            tpoly = box(c[0] - ext[0] / 2 - 0.3, c[1] - ext[1] / 2 - 0.3, c[0] + ext[0] / 2 + 0.3, c[1] + ext[1] / 2 + 0.3)
            if g.contains(tpoly):
                body = body - mf.Manifold.extrude(cs, ENGR + 0.2).translate((0, 0, -PEG_H - 0.2))
                label = hgt
                break
    pc.update(body=body, footprint=fp, holes=holes_xy, cav_vol=cav_vol, label=label)
    return pc


def base_outline(pieces):
    allfp = unary_union([p["footprint"] for p in pieces])
    o = allfp.buffer(7.0).buffer(9.0).buffer(-9.0)
    o = unary_union([o, allfp.convex_hull.buffer(4.0)]).buffer(3).buffer(-3)
    return o.simplify(0.05)


def build_base(pieces, eye):
    out = base_outline(pieces)
    base = prism(out, -BASE_T, -0.8) + prism(out.buffer(-0.8), -0.8, 0.0)
    # метка направления на глаз — V-вырез в переднем крае
    ymin = out.bounds[1]
    notch = Polygon([(eye[0] - 2.2, ymin - 1), (eye[0] + 2.2, ymin - 1), (eye[0], ymin + 2.2)])
    base = base - prism(notch, -BASE_T - 1, 1)
    for p in pieces:
        pocket = p["footprint"].buffer(CLEAR - 0.05, quad_segs=8)
        base = base - prism(pocket, -PEG_H, 0.5)
        # номер на дне гнезда
        g = p["footprint"].buffer(-0.8)
        if g.is_empty:
            continue
        g = max(g.geoms, key=lambda q: q.area) if hasattr(g, "geoms") else g
        c = np.array(polylabel(g, 0.05).coords[0])
        for hgt in (4.0, 3.2, 2.6, 2.0):
            cs, ext = text_cs(str(p["num"]), hgt, c)
            tpoly = box(c[0] - ext[0] / 2 - 0.3, c[1] - ext[1] / 2 - 0.3, c[0] + ext[0] / 2 + 0.3, c[1] + ext[1] / 2 + 0.3)
            if g.contains(tpoly):
                base = base - mf.Manifold.extrude(cs, ENGR + 0.1).translate((0, 0, -PEG_H - ENGR))
                break
    # подпись на дне основания
    b = out.bounds
    msg = [f"ГЛАЗ x{eye[0]:+.0f} y{eye[1]:+.0f} h{eye[2] + BASE_T:.0f}", "мм, h — от стола"]
    for k, line in enumerate(msg):
        cs, ext = text_cs(line, 4.0, np.array([(b[0] + b[2]) / 2, b[1] + 14 - 7 * k]), mirror=True)
        base = base - mf.Manifold.extrude(cs, ENGR + 0.2).translate((0, 0, -BASE_T - 0.2))
    return base, out


def build_viewer(cam):
    """Стойка-глазок: узкое место отверстия Ø2.5 ровно в точке обзора, раструб под глаз."""
    E, f = cam.E, cam.f
    zt = -BASE_T  # стол
    ang = np.degrees(np.arctan2(f[1], f[0]))
    pitch = np.degrees(np.arcsin(f[2]))
    # локальные оси: X — к сцене, Y — влево, Z — вверх; начало — под глазом
    head = mf.Manifold.cylinder(2.4, 9.0, 9.0, 72, True)
    head = head - mf.Manifold.cylinder(8.0, 1.25, 1.25, 48, True)
    cone = mf.Manifold.cylinder(1.6, 6.0, 1.25, 64).translate((0, 0, -1.3))  # раструб со стороны глаза (−z)
    head = (head - cone).translate((0, 0, -0.7))  # узкое место (z 0.2..1.2) → центр в 0
    head = head.rotate((0, 90 - pitch, 0)).translate((0, 0, E[2]))
    foot = mf.Manifold.cube((34.0, 40.0, 3.0), True).translate((-7.0, 0, zt + 1.5))
    col_h = (E[2] - 7.5) - (zt + 3.0)
    col = mf.Manifold.cube((6.0, 10.0, col_h), True).translate((0, 0, zt + 3.0 + col_h / 2))
    tri = mf.CrossSection([np.array([[0, 0], [0, 22.0], [-16.0, 0]])], mf.FillRule.EvenOdd)
    gus = mf.Manifold.extrude(tri, 2.4).rotate((90, 0, 0))  # треугольник в плоскости XZ, толщина по −Y
    gussets = gus.translate((-3.0, 1.2 + 2.4 + 1.0, zt + 3.0)) + gus.translate((-3.0, -1.2 - 1.0 + 2.4 - 2.4, zt + 3.0))
    cs, _ = text_cs("ГЛАЗ", 5.0, np.array([-17.0, 0.0]), angle=-np.pi / 2)
    v = head + col + foot + gussets
    v = v - mf.Manifold.extrude(cs, ENGR + 0.1).translate((0, 0, zt + 3.0 - ENGR))
    return v.rotate((0, 0, ang)).translate((E[0], E[1], 0))


def main(out_full, out_parts):
    os.makedirs(out_full, exist_ok=True)
    os.makedirs(out_parts, exist_ok=True)
    R = B.compute()
    cam = R["cam"]
    pieces = make_pieces(R)
    for pc in pieces:
        build_piece(R, pc)
    # проверка пересечений деталей
    bad = []
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            if pieces[a]["footprint"].distance(pieces[b]["footprint"]) > 5:
                continue
            v = (pieces[a]["body"] ^ pieces[b]["body"]).volume()
            if v > 1e-4:
                bad.append((pieces[a]["num"], pieces[b]["num"], round(v, 4)))
                # срезаем у второй детали всё, что лезет в первую (+ зазор 0.05)
                pieces[b]["body"] = pieces[b]["body"] - pieces[a]["body"].refine_to_length(50)
    print("пересечения деталей (исправлены):", bad or "нет")
    base, outline = build_base(pieces, cam.E)
    viewer = build_viewer(cam)
    info = []
    for pc in pieces:
        name = f"{pc['num']:02d}_{pc['kind']}{'_moss' if pc['moss'] else ''}.stl"
        t = save(pc["body"], os.path.join(out_parts, name))
        ext = t.extents
        info.append(dict(num=pc["num"], file=name, kind=pc["kind"], moss=pc["moss"],
                         size=[round(float(e), 1) for e in ext], volume_ml=round(t.volume / 1000, 2),
                         hollow=pc["cav_vol"] > 0, drains=len(pc["holes"]), watertight=bool(t.is_watertight)))
    tb = save(base, os.path.join(out_parts, "00_base.stl"))
    tv = save(viewer, os.path.join(out_parts, "viewer_post.stl"))
    info.append(dict(num=0, file="00_base.stl", size=[round(float(e), 1) for e in tb.extents],
                     volume_ml=round(tb.volume / 1000, 2), watertight=bool(tb.is_watertight)))
    info.append(dict(num=-1, file="viewer_post.stl", size=[round(float(e), 1) for e in tv.extents],
                     volume_ml=round(tv.volume / 1000, 2), watertight=bool(tv.is_watertight)))
    # цельная версия: основание без гнёзд + камни + мох
    plate = prism(outline, -BASE_T, -0.8) + prism(outline.buffer(-0.8), -0.8, 0.0)
    rocks = mf.Manifold.batch_boolean([convex_manifold(s.clip_z(-1.0).grow(0.01)) for s in R["vis"]], mf.OpType.Add)
    moss = mf.Manifold.batch_boolean(list(R["pads"].values()), mf.OpType.Add)
    full = plate + rocks + moss
    tf = save(full, os.path.join(out_full, "moss_line_rocks_full.stl"))
    # 3MF на 3 цвета: вода / камни / мох (мох вдавлен в камень на 0.8 мм — для FDM)
    moss_deep = []
    for k in R["pads"]:
        s = R["vis"][k]
        m = moss_solid(cam, s, R["regions"][k], front_facets(cam, s), sink=0.8)
        if m is not None:
            moss_deep.append(m)
    moss3 = mf.Manifold.batch_boolean(moss_deep, mf.OpType.Add)
    rocks3 = mf.Manifold.batch_boolean([convex_manifold(s.clip_z(0.0).grow(0.01)) for s in R["vis"]], mf.OpType.Add) - moss3
    water3 = plate
    sc = trimesh.Scene()
    for nm, m, col in (("water", water3, [61, 134, 173, 255]), ("rocks", rocks3, [216, 199, 161, 255]), ("moss", moss3, [91, 154, 46, 255])):
        t = to_tm(m)
        t.visual.face_colors = col
        sc.add_geometry(t, geom_name=nm)
    sc.export(os.path.join(out_full, "moss_line_rocks_3color.3mf"))
    info_full = dict(full_size=[round(float(e), 1) for e in tf.extents], full_volume_ml=round(tf.volume / 1000, 1),
                     full_watertight=bool(tf.is_watertight))
    json.dump(dict(parts=info, full=info_full, eye=cam.E.tolist(), look=SC.LOOK.tolist(),
                   base_thickness=BASE_T, clear=CLEAR, peg_h=PEG_H),
              open(os.path.join(out_parts, "_info.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info_full, ensure_ascii=False))
    for i in info:
        print(i)
    return R, pieces, base, viewer, outline


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
