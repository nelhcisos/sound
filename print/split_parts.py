"""GLB -> несколько STL-частей для печати, со штифтами-соединителями.
usage: python3 split_parts.py in.glb out_dir name height_mm z_cut1 z_cut2 ...
Разрезы — высоты в мм от низа (после масштабирования до height_mm).
Нижняя часть получает штифт, верхняя — гнездо с зазором.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

src, out, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
H = float(sys.argv[4])
CUTS = sorted(float(z) for z in sys.argv[5:])
PITCH = 0.1        # мм
PEG_R_MAX = 2.5    # радиус штифта, мм
PEG_H = 3.0        # высота штифта, мм
GAP = 0.15         # зазор гнезда, мм
out.mkdir(parents=True, exist_ok=True)

s = trimesh.load(src, force="scene")
m = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m.apply_scale(H / m.extents[2])
m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
print("размер мм:", m.extents.round(1))

# сплошное тело из вокселей (как в hollow.py)
vox = m.voxelized(PITCH)
grid = vox.matrix.copy()
D = 6
grid = np.pad(grid, D + 2)
dil = ndimage.binary_dilation(grid, iterations=D)
filled = ndimage.binary_fill_holes(dil)
solid = ndimage.binary_erosion(filled, iterations=D) | (grid & filled)
solid = solid[D + 2:-(D + 2), D + 2:-(D + 2), D + 2:-(D + 2)]
origin = vox.transform[:3, 3]
z_of = lambda zmm: int(round((zmm - origin[2]) / PITCH))
print("воксели:", solid.shape)

X, Y = np.ogrid[: solid.shape[0], : solid.shape[1]]
parts = []
edges = [0] + [z_of(z) for z in CUTS] + [solid.shape[2]]
for a, b in zip(edges[:-1], edges[1:]):
    p = np.zeros_like(solid)
    p[:, :, a:b] = solid[:, :, a:b]
    parts.append(p)

peg_h = int(round(PEG_H / PITCH))
gap = GAP / PITCH
for i, zc in enumerate(edges[1:-1]):
    lower, upper = parts[i], parts[i + 1]
    # стык: пересечение нижнего слоя верха и верхнего слоя низа
    face = solid[:, :, zc - 1] & solid[:, :, zc]
    lab, k = ndimage.label(face)
    for j in range(1, k + 1):
        comp = lab == j
        if comp.sum() * PITCH ** 2 < 20:  # мелкие куски — без штифта
            continue
        dist = ndimage.distance_transform_edt(comp)
        cx, cy = np.unravel_index(np.argmax(dist), dist.shape)
        r = min(PEG_R_MAX / PITCH, dist.max() * 0.5)
        h = min(peg_h, int((upper.any(axis=(0, 1)).nonzero()[0].max() - zc) * 0.6))
        disk = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        hole = (X - cx) ** 2 + (Y - cy) ** 2 <= (r + gap) ** 2
        lower[:, :, zc:zc + h] |= disk[:, :, None]
        upper[:, :, zc:zc + h + 3] &= ~hole[:, :, None]
        print(f"стык {i + 1}: штифт Ø{2 * r * PITCH:.1f} мм × {h * PITCH:.1f} мм")

names = sys.argv[5 + len(CUTS):] or None
for i, p in enumerate(parts):
    f = ndimage.gaussian_filter(np.pad(p, 2).astype(np.float32), 0.8)
    v, faces, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 2) * PITCH + origin
    v[:, 2] -= v[:, 2].min()  # каждая часть стоит на столе
    mesh = trimesh.Trimesh(v, faces)
    mesh.fix_normals()
    fn = out / f"{name}_{i + 1}.stl"
    mesh.export(fn)
    print(fn.name, mesh.extents.round(1), "замкнута:", mesh.is_watertight)
