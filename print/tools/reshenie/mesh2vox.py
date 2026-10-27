"""метки граней (L.npy) -> воксели мастера 100 мм -> labels.npz (формат finalize.py) + master_100mm/*.stl"""
import sys, os, gc, numpy as np, trimesh
from scipy import ndimage
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_front
GLB, LNPY, OUT = sys.argv[1:4]
P = float(os.environ.get("PITCH", 0.1)); H = 100.0
NAMES = ["platform", "shoe_left", "shoe_right", "pant_left", "pant_right", "body",
         "sleeve_left", "sleeve_right", "hands", "neck", "head", "hair"]
ID = {n: i + 1 for i, n in enumerate(NAMES)}
os.makedirs(OUT, exist_ok=True)
if os.environ.get("STAGE2"):
    lab = np.load(os.path.join(OUT, "_lab_raw.npy")); meta = np.load(os.path.join(OUT, "_meta.npy"))
    origin = meta[:3]; kz = int(meta[3]); SH = lab.shape
else:
  m, _ = load_front(GLB, H); L = np.load(LNPY)
  lo3 = m.bounds[0] - P
  shape3 = np.ceil((m.bounds[1] - m.bounds[0]) / P).astype(int) + 3
  surf = np.zeros(shape3, np.uint8)
  tri = m.triangles; ar = m.area_faces
  for i0 in range(0, len(tri), 200000):
      t = tri[i0:i0 + 200000]; a = ar[i0:i0 + 200000]; lb = L[i0:i0 + 200000]
      n = np.maximum(1, np.ceil(a / (P * 0.5) ** 2).astype(int))
      reps = np.repeat(np.arange(len(t)), n)
      u = np.random.rand(len(reps), 2); f = u.sum(1) > 1; u[f] = 1 - u[f]
      tt = t[reps]
      pts = tt[:, 0] + u[:, :1] * (tt[:, 1] - tt[:, 0]) + u[:, 1:] * (tt[:, 2] - tt[:, 0])
      ij = np.floor((pts - lo3) / P).astype(np.int32)
      surf[ij[:, 0], ij[:, 1], ij[:, 2]] = lb[reps]
  del tri, ar; gc.collect()
  D = max(2, int(round(0.6 / P)))
  g = np.pad(surf > 0, D + 2)
  fz = ndimage.binary_fill_holes(ndimage.binary_dilation(g, iterations=D))
  solid = ndimage.binary_erosion(fz, iterations=D) | (g & fz)
  del fz, g; gc.collect()
  solid = solid[D + 2:-(D + 2), D + 2:-(D + 2), D + 2:-(D + 2)]
  pad = 3
  solid = np.pad(solid, pad); surf = np.pad(surf, pad)
  origin = lo3 + P / 2 - pad * P
  SH = solid.shape; print("воксели", SH, "%.0f см³" % (solid.sum() * P**3 / 1000))
  # внутренность: ближайшая метка поверхности на грубой сетке
  F = 4
  cs = tuple(s // F * F for s in SH)
  sb = surf[:cs[0], :cs[1], :cs[2]].reshape(cs[0] // F, F, cs[1] // F, F, cs[2] // F, F).max((1, 3, 5))
  idx = ndimage.distance_transform_edt(sb == 0, return_distances=False, return_indices=True)
  cl = sb[idx[0], idx[1], idx[2]]; del idx, sb; gc.collect()
  lab = np.zeros(SH, np.uint8)
  up = np.repeat(np.repeat(np.repeat(cl, F, 0), F, 1), F, 2)
  lab[:cs[0], :cs[1], :cs[2]] = up; del up
  lab[cs[0]:] = lab[cs[0] - 1:cs[0]] if cs[0] < SH[0] else lab[cs[0]:]
  s_ = surf > 0
  lab[s_] = surf[s_]; del surf, s_
  lab[~solid] = 0
  # толщина волос: глубже 6 мм от поверхности -> голова
  dc = ndimage.distance_transform_edt(solid[::2, ::2, ::2]).astype(np.float32) * (2 * P)
  w = np.nonzero(lab == ID["hair"]); dd = dc[w[0] // 2, w[1] // 2, w[2] // 2]
  k = dd > 6.0; lab[w[0][k], w[1][k], w[2][k]] = ID["head"]; del w, dd, k, dc; gc.collect()
  # платформа строго ниже верха платформы
  z = origin[2] + np.arange(SH[2]) * P
  C = m.triangles_center; up_ = (m.face_normals[:, 2] > 0.95) & (C[:, 2] < 15)
  hz = np.histogram(C[up_, 2], bins=np.arange(0, 15, 0.1)); ztop = hz[1][np.argmax(hz[0])] + 0.05
  kz = int(np.searchsorted(z, ztop))
  sub = lab[:, :, :kz]; sub[sub > 0] = ID["platform"]
  abv = lab[:, :, kz:]; bad = abv == ID["platform"]
  if bad.any():
      rep = ndimage.grey_dilation(np.where(bad, 0, abv), size=(1, 1, 7))
      abv[bad] = rep[bad]
  print("верх платформы z=%.2f" % ztop)
  RAW = os.path.join(OUT, "_lab_raw.npy")
  np.save(RAW, lab); np.save(os.path.join(OUT, "_meta.npy"), np.r_[origin, kz])
  del solid; gc.collect()
# ---------- этап 2: сглаживание (экономно по памяти) ----------
solid = lab > 0
sig = 0.5 / P
score = np.zeros(SH, np.uint8); best = lab.copy()
for lid in range(1, len(NAMES) + 1):
    objs = ndimage.find_objects(lab, max_label=lid)
    o = objs[lid - 1] if len(objs) >= lid else None
    if o is None: continue
    sl = tuple(slice(max(s.start - 8, 0), s.stop + 8) for s in o)
    gg = ndimage.gaussian_filter((lab[sl] == lid).astype(np.float32), sig)
    g8 = np.round(gg * 255).astype(np.uint8); del gg
    sc = score[sl]; bl = best[sl]; u_ = g8 > sc; sc[u_] = g8[u_]; bl[u_] = lid
    del g8, u_; gc.collect()
lab = np.where(solid, best, 0).astype(np.uint8); del score, best; gc.collect()
low = lab[:, :, :kz]; low[low > 0] = ID["platform"]
abv = lab[:, :, kz:]; abv[abv == ID["platform"]] = ID["shoe_left"]
# у каждой детали одно тело (в рамке детали)
for it in range(2):
    for lid in range(1, len(NAMES) + 1):
        objs = ndimage.find_objects(lab, max_label=lid)
        o = objs[lid - 1] if len(objs) >= lid else None
        if o is None: continue
        sl = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in o)
        sub = lab[sl]
        lb, n = ndimage.label(sub == lid)
        if n <= 1: del lb; continue
        sz = ndimage.sum(sub == lid, lb, range(1, n + 1)); ob = ndimage.find_objects(lb)
        for j in np.argsort(sz)[::-1][1:]:
            bb = tuple(slice(max(q.start - 3, 0), q.stop + 3) for q in ob[j])
            comp = lb[bb] == (j + 1)
            ring = ndimage.binary_dilation(comp, iterations=2) & (sub[bb] > 0) & ~comp
            vals = sub[bb][ring]; vals = vals[(vals != lid) & (vals > 0)]
            if len(vals): sub[bb][comp] = np.bincount(vals).argmax()
        del lb; gc.collect()
np.savez_compressed(os.path.join(OUT, "labels.npz"), lab=lab, origin=origin, pitch=P,
                    ref_height=H, names=np.array(NAMES))
md = os.path.join(OUT, "master_100mm"); os.makedirs(md, exist_ok=True)
for lid, n in enumerate(NAMES, 1):
    w = np.nonzero(lab == lid)
    if not len(w[0]): continue
    sl = tuple(slice(max(a.min() - 3, 0), a.max() + 4) for a in w)
    sub = np.pad(lab[sl] == lid, 1).astype(np.float32)
    v, fc, _, _ = measure.marching_cubes(ndimage.gaussian_filter(sub, 0.6), 0.5)
    v = (v - 1 + [q.start for q in sl]) * P + origin
    t = trimesh.Trimesh(v, fc, process=True)
    if t.volume < 0: t.invert()
    t.export(os.path.join(md, n + ".stl"))
    print("мастер %-13s %s мм  %.1f см³ тел=%d" % (n, t.extents.round(1), t.volume / 1000, len(t.split(only_watertight=False))))
