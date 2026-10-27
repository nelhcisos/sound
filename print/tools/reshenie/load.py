"""Загрузка GLB с текстурой (Z вверх, лицо -> -Y, масштаб в мм), цвет граней, ортопроекции-«сплэты» для превью и выбора семян."""
import numpy as np, trimesh
def load(path, H=100.0):
    s = trimesh.load(path, force="scene")
    g = list(s.geometry.values())[0]
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"), float) / 255
    uv = g.visual.uv
    m = trimesh.Trimesh(g.vertices, g.faces, process=False)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    sc = H / m.extents[2]
    m.apply_scale(sc)
    m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
    fuv = uv[m.faces].mean(1)
    th, tw = tex.shape[:2]
    px = np.clip((fuv[:, 0] % 1) * (tw - 1), 0, tw - 1).astype(int)
    py = np.clip((1 - fuv[:, 1] % 1) * (th - 1), 0, th - 1).astype(int)
    fcol = tex[py, px]
    nf = len(m.faces); m.merge_vertices(merge_tex=True, merge_norm=True); assert len(m.faces) == nf
    return m, fcol

def splat(m, vals, view="front", res=0.2):
    """ортопроекция по центрам граней. front: камера с -Y; back: с +Y; side: с +X"""
    c = m.triangles_center; n = m.face_normals
    if view == "front": u, v, d, nn = c[:, 0], c[:, 2], c[:, 1], -n[:, 1]
    elif view == "back": u, v, d, nn = -c[:, 0], c[:, 2], -c[:, 1], n[:, 1]
    elif view == "left": u, v, d, nn = -c[:, 1], c[:, 2], c[:, 0], -n[:, 0]
    else: u, v, d, nn = c[:, 1], c[:, 2], -c[:, 0], n[:, 0]
    iu = ((u - u.min()) / res).astype(int); iv = ((v.max() - v) / res).astype(int)
    W, Hh = iu.max() + 1, iv.max() + 1
    key = iv * W + iu
    o = np.lexsort((d, key)); k = key[o]; first = np.r_[True, k[1:] != k[:-1]]
    sel = o[first]
    img = np.ones((Hh * W, 3)); 
    shade = np.clip(0.35 + 0.65 * np.clip(nn, 0, 1), 0, 1)[:, None]
    img[key[sel]] = vals[sel] * shade[sel] if vals.ndim == 2 else vals[sel]
    idx = np.full(Hh * W, -1); idx[key[sel]] = sel
    return img.reshape(Hh, W, 3), idx.reshape(Hh, W)

def load_front(path, H=100.0):
    m, fc = load(path, H)
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1]))  # лицо -> -Y
    return m, fc

def classify(m, fc):
    import colorsys
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in fc[:: 1]]) if len(fc) < 0 else None
    r, g, b = fc.T
    mx, mn = fc.max(1), fc.min(1)
    v = mx; s = (mx - mn) / np.maximum(mx, 1e-6)
    z = m.triangles_center[:, 2]
    cls = np.full(len(fc), 2, np.uint8)            # 2 чёрный (одежда)
    red = (r > 0.45) & (r > g * 1.6) & (r > b * 1.6) & (s > 0.45)
    skin = (v > 0.55) & (r >= g) & (g >= b * 0.95) & (s > 0.12) & (s < 0.55) & ~red
    cls[red] = 1; cls[skin] = 3
    return cls
