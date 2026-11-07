"""Общие функции разборки v3: сетка <-> manifold, плоскости, локальные резаки, рендер."""
import numpy as np, trimesh
import manifold3d as m3


# ---------- сетки ----------
def to_man(tm):
    mesh = m3.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                   tri_verts=np.asarray(tm.faces, np.uint32))
    man = m3.Manifold(mesh)
    if man.status() != m3.Error.NoError:
        raise RuntimeError("сетка не manifold: %s" % man.status())
    return man


def to_tm(man):
    mm = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def frame(n):
    """ортонормированный базис (u, v, n)"""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, n); u /= np.linalg.norm(u); v = np.cross(n, u)
    return u, v, n


def mat34(origin, u, v, w):
    """матрица 3x4: локальные X,Y,Z -> u,v,w, начало в origin"""
    return np.c_[np.c_[u, v, w], origin].astype(float)


def vec(x):
    if isinstance(x, str):
        s = -1.0 if x.startswith("-") else 1.0
        return s * np.eye(3)["xyz".index(x[-1])]
    x = np.asarray(x, float); return x / np.linalg.norm(x)


# ---------- резак ----------
def _region(spec, lo, hi, body_tm=None):
    """одна выпуклая-ish область: пересечение полупространств (нормаль ВНУТРЬ детали)
    + цилиндр / коробка / шар / призма по сечению тела"""
    c = m3.Manifold.cube(hi - lo).translate(lo)
    for pl in spec.get("planes", []):
        n = vec(pl["n"]); p = np.asarray(pl["p"], float)
        c = c.trim_by_plane(n, float(p @ n))
    if "cyl" in spec:
        cy = spec["cyl"]; n0 = vec(spec["planes"][0]["n"]) if spec.get("planes") else np.array([0, 0, 1.0])
        ax = vec(cy.get("axis", n0)); u, v, w = frame(ax)
        ctr = np.asarray(cy["c"], float); L = float(cy.get("len", 600))
        cyl = m3.Manifold.cylinder(L, float(cy["r"]), float(cy["r"]), 96).translate([0, 0, -L / 2])
        c = c ^ cyl.transform(mat34(ctr, u, v, w))
    if "box" in spec:
        b0 = np.asarray(spec["box"][0], float); b1 = np.asarray(spec["box"][1], float)
        c = c ^ m3.Manifold.cube(b1 - b0).translate(b0)
    if "sphere" in spec:
        sp = spec["sphere"]; c = c ^ m3.Manifold.sphere(float(sp["r"]), 96).translate(sp["c"])
    if "prism" in spec:
        c = c ^ prism_region(spec["prism"], body_tm, lo, hi)
    if "ellipsoid" in spec:
        # «шлем» по форме черепа: снаружи (outside: true) или внутри эллипсоида, раздутого на offset мм
        e = spec["ellipsoid"]; r = np.asarray(e["r"], float) + float(e.get("offset", 0.0))
        el = m3.Manifold.sphere(1.0, 192).scale(r).translate(e["c"])
        c = (c - el) if e.get("outside", True) else (c ^ el)
    if "mesh" in spec:
        # внутри заданной сетки (например, старая деталь по цвету), раздутой масштабом от её центра
        me = spec["mesh"]; t = trimesh.load(me["path"], force="mesh")
        if "translate" in me: t.apply_translation(me["translate"])
        k = float(me.get("scale", 1.0)); cc = t.bounding_box.centroid
        t.apply_translation(-cc); t.apply_scale(k); t.apply_translation(cc)
        c = c ^ to_man(t)
    for sub in spec.get("minus", []):        # вычесть области (например, ногу внутри ботинка)
        c = c - _region(sub, lo, hi, body_tm)
    return c


def prism_region(pr, body_tm, lo, hi):
    """призма: сечение тела в точке "at" поперёк оси "axis", контур "pick" (ближайший к точке) или все,
    раздутый на "margin" мм, вытянутый вдоль оси через всю модель — резак «по форме конечности»"""
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    ax = vec(pr.get("axis", "z")); at = np.asarray(pr["at"], float); u, v, w = frame(ax)
    sec = body_tm.section(plane_origin=at, plane_normal=ax)
    if sec is None: raise SystemExit("prism: нет сечения в %s" % at)
    polys = [Polygon(np.c_[(e - at) @ u, (e - at) @ v]).buffer(0) for e in sec.discrete if len(e) >= 4]
    shape = Polygon()
    for P in sorted(polys, key=lambda P: -P.area): shape = shape.symmetric_difference(P)
    geoms = list(getattr(shape, "geoms", [shape]))
    if "pick" in pr:
        pk = np.asarray(pr["pick"], float); q = Point((pk - at) @ u, (pk - at) @ v)
        geoms = [min(geoms, key=lambda g: g.distance(q))]
    g = unary_union(geoms).buffer(float(pr.get("margin", 0.4)), join_style=1)
    g = g.simplify(0.05)
    contours = []
    for gg in getattr(g, "geoms", [g]):
        contours.append(np.asarray(gg.exterior.coords)[:-1])
        for r in gg.interiors: contours.append(np.asarray(r.coords)[:-1])
    cs = m3.CrossSection(contours, m3.FillRule.EvenOdd)
    L = float(np.linalg.norm(hi - lo)) * 2 + 100
    return m3.Manifold.extrude(cs, L).translate([0, 0, -L / 2]).transform(mat34(at, u, v, w))


def cutter(spec, bbox_lo, bbox_hi, body_tm=None):
    """Область детали. spec: одна область (planes/cyl/box/sphere/prism) или "any": [области] — объединение."""
    pad = 50.0
    lo = np.asarray(bbox_lo) - pad; hi = np.asarray(bbox_hi) + pad
    regs = spec["any"] if "any" in spec else [spec]
    c = None
    for r in regs:
        x = _region(r, lo, hi, body_tm)
        c = x if c is None else c + x
    return c


def joint_plane(spec):
    """плоскость стыка = первая плоскость первой области"""
    r = spec["any"][0] if "any" in spec else spec
    return r["planes"][0]


# ---------- рендер ----------
PAL = [(230, 75, 60), (60, 140, 230), (250, 200, 40), (80, 190, 110), (170, 90, 210), (250, 140, 40),
       (40, 190, 200), (230, 110, 170), (140, 110, 80), (120, 120, 130), (200, 230, 90), (90, 90, 200),
       (250, 160, 140), (60, 110, 70), (200, 60, 120), (160, 200, 240), (240, 220, 170), (100, 60, 40)]
ROT = {"front": np.eye(3), "left": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
       "back": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), "right": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
       "top": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), "bottom": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])}


def render(parts, views=("front", "left", "back", "right"), explode=0.0, W=420, grid=None, scale=None, center=None):
    """parts: [(name, trimesh, rgb)] -> список PIL-картинок. Камера спереди смотрит с -Y."""
    from PIL import Image, ImageDraw
    L = np.array([0.35, -0.8, 0.5]); L /= np.linalg.norm(L)
    S = []
    for name, m, col in parts:
        k = int(max(100000, min(2500000, m.area * 160)))
        p, fi = trimesh.sample.sample_surface(m, k)
        col = np.asarray(col, float)
        S.append((p, m.face_normals[fi], col[fi] if col.ndim == 2 else col, m.centroid))
    allp = np.vstack([s[0] for s in S])
    C = (allp.min(0) + allp.max(0)) / 2 if center is None else np.asarray(center, float)
    C = C.copy(); zmin = allp[:, 2].min()
    ext = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), np.ptp(allp[:, 2])) * (1 + explode * 0.9)
    sc = (W - 30) / ext if scale is None else scale
    out = []
    for vname in views:
        R = ROT[vname]; Q = []; Cc = []
        for p, nr, col, cen in S:
            off = (cen - C) * np.array([1, 1, 0.8]) * explode
            q = (p + off - C) @ R.T; nn = nr @ R.T
            sh = np.clip(nn @ L, 0, 1) * 0.65 + 0.35
            Q.append(q); Cc.append(sh[:, None] * col if col.ndim == 2 else np.outer(sh, col))
        q = np.vstack(Q); cc = np.vstack(Cc)
        x = (q[:, 0] * sc + W / 2).astype(int); y = (W / 2 - q[:, 2] * sc).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < W)
        x, y, d, cc = x[ok], y[ok], q[ok, 1], cc[ok]
        o = np.lexsort((d, y * W + x)); key = (y * W + x)[o]; first = np.r_[True, key[1:] != key[:-1]]
        img = np.full((W * W, 3), 255, np.uint8); img[key[first]] = np.clip(cc[o][first], 0, 255).astype(np.uint8)
        im = Image.fromarray(img.reshape(W, W, 3)); dr = ImageDraw.Draw(im)
        if grid:  # горизонтальные линии уровней z (мм) — для выбора плоскостей
            zc = C[2] if vname not in ("top", "bottom") else 0
            for z in np.arange(np.floor(allp[:, 2].min() / grid) * grid, allp[:, 2].max() + grid, grid):
                yy = int(W / 2 - (z - zc) * sc)
                if 0 <= yy < W:
                    dr.line([(0, yy), (12, yy)], fill=(200, 0, 0)); dr.text((14, yy - 6), "%g" % z, fill=(200, 0, 0))
            ax_sg = {"front": (0, 1), "back": (0, -1), "left": (1, 1), "right": (1, -1)}.get(vname)
            if ax_sg is not None:
                xs_axis, sgn = ax_sg
                for xv in np.arange(-60, 61, grid):
                    xx = int(W / 2 + sgn * (xv - C[xs_axis]) * sc)
                    if 0 <= xx < W:
                        dr.line([(xx, W - 10), (xx, W)], fill=(0, 0, 200)); dr.text((xx - 6, W - 22), "%g" % xv, fill=(0, 0, 200))
        dr.text((W - 70, 4), vname, fill=(0, 0, 0))
        out.append(im)
    return out


def sheet(images, ncol, legend=None, W=420):
    from PIL import Image, ImageDraw
    nrow = (len(images) + ncol - 1) // ncol
    lw = 170 if legend else 0
    o = Image.new("RGB", (W * ncol + lw, max(W * nrow, 22 * len(legend or []) + 20)), "white")
    for i, im in enumerate(images):
        o.paste(im, ((i % ncol) * W, (i // ncol) * W))
    if legend:
        d = ImageDraw.Draw(o)
        for i, (n, col) in enumerate(legend):
            d.rectangle([W * ncol + 8, 10 + i * 22, W * ncol + 26, 26 + i * 22], fill=tuple(int(c) for c in col))
            d.text((W * ncol + 32, 12 + i * 22), n, fill=(0, 0, 0))
    return o
