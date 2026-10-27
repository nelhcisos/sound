"""Шея — только видимое кольцо (±2 мм), остальное внутри -> тело/голова. usage: fixneck.py cutdir"""
import numpy as np, sys, os, trimesh
from scipy import ndimage
from skimage import measure
D = sys.argv[1]
d = dict(np.load(os.path.join(D, "labels.npz"))); lab = d["lab"]; P = float(d["pitch"]); org = d["origin"]
NAMES = [str(x) for x in d["names"]]; ID = {n: i + 1 for i, n in enumerate(NAMES)}
z = org[2] + np.arange(lab.shape[2]) * P
nk = lab == ID["neck"]
# низ видимой шеи: самый нижний воксель шеи на поверхности
outer = nk & ~ndimage.binary_erosion(lab > 0)
zs = np.nonzero(outer.any((0, 1)))[0]
zlow = z[zs[0]] - 2.0
k = int(np.searchsorted(z, zlow))
sub = lab[:, :, :k]; sub[sub == ID["neck"]] = ID["body"]
ztop = z[zs[-1]] + 2.0; k2 = int(np.searchsorted(z, ztop))
sub = lab[:, :, k2:]; sub[sub == ID["neck"]] = ID["head"]
print("верх видимой шеи z=%.1f" % z[zs[-1]])
print("шея: низ видимой части z=%.1f, обрезка ниже %.1f" % (z[zs[0]], zlow))
d["lab"] = lab; np.savez_compressed(os.path.join(D, "labels.npz"), **d)
for n in ["neck", "body", "head"]:
    lid = ID[n]; w = np.nonzero(lab == lid)
    sl = tuple(slice(max(a.min() - 3, 0), a.max() + 4) for a in w)
    s2 = np.pad(lab[sl] == lid, 1).astype(np.float32)
    v, fc, _, _ = measure.marching_cubes(ndimage.gaussian_filter(s2, 0.6), 0.5)
    t = trimesh.Trimesh((v - 1 + [q.start for q in sl]) * P + org, fc, process=True)
    if t.volume < 0: t.invert()
    t.export(os.path.join(D, "master_100mm", n + ".stl")); print(n, t.extents.round(1), "%.1f см³" % (t.volume / 1000), len(t.split(only_watertight=False)))
