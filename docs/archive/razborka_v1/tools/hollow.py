"""GLB -> полая STL заданной высоты.
usage: python3 hollow.py in.glb out.stl [height_mm=40] [wall_mm=1.5] [drain_mm=2.5]
Модель вокселизуется (заодно закрываются дыры), стенка = эрозия внутрь,
в дне — сливное отверстие для смолы.
"""
import sys
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

src, dst = sys.argv[1], sys.argv[2]
H = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
WALL = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
DRAIN = float(sys.argv[5]) if len(sys.argv) > 5 else 2.5
PITCH = 0.1  # мм

s = trimesh.load(src, force="scene")
m = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
# glb Y-up -> Z-up
m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m.apply_scale(H / m.extents[2])
m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
print("размер мм:", m.extents.round(1), "граней:", len(m.faces), "замкнута:", m.is_watertight)

# вокселизация поверхности, затем заливка внутренности
vox = m.voxelized(PITCH)
grid = vox.matrix.copy()
D = 6  # временное утолщение, чтобы закрыть дыры в сетке
grid = np.pad(grid, D + 2)
dil = ndimage.binary_dilation(grid, iterations=D)
filled = ndimage.binary_fill_holes(dil)
solid = ndimage.binary_erosion(filled, iterations=D) | (grid & filled)
solid = solid[D + 2:-(D + 2), D + 2:-(D + 2), D + 2:-(D + 2)]
print("воксели:", solid.shape)

# полость: эрозия на толщину стенки
n = int(round(WALL / PITCH))
ball = ndimage.generate_binary_structure(3, 1)
inner = ndimage.binary_erosion(solid, structure=ball, iterations=n)
# убираем мелкие куски полости (тонкие руки и т.п. остаются сплошными)
lab, k = ndimage.label(inner)
if k:
    sizes = ndimage.sum(inner, lab, range(1, k + 1))
    keep = 1 + int(np.argmax(sizes))
    inner = lab == keep
shell = solid & ~inner

# сливное отверстие снизу в центре полости
if inner.any():
    zs = np.where(inner.any(axis=(0, 1)))[0]
    z0 = zs[0]
    # точка полости в нижнем слое, ближайшая к её центру (не в зазоре между ногами)
    xs, ys = np.nonzero(inner[:, :, z0])
    mx, my = xs.mean(), ys.mean()
    i = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
    cx, cy = int(xs[i]), int(ys[i])
    r = DRAIN / 2 / PITCH
    X, Y = np.ogrid[: shell.shape[0], : shell.shape[1]]
    disk = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    shell[:, :, : z0 + 1][disk] = False
    print("полость: объём %.1f мм³, отверстие Ø%.1f мм" % (inner.sum() * PITCH**3, DRAIN))
else:
    print("полость не поместилась — модель сплошная")

pad = np.pad(shell, 1)
field = ndimage.gaussian_filter(pad.astype(np.float32), 0.8)
verts, faces, _, _ = measure.marching_cubes(field, 0.5)
verts = (verts - 1) * PITCH + vox.translation
out = trimesh.Trimesh(verts, faces, process=True)
if out.volume < 0:
    out.invert()

out.apply_translation([0, 0, -out.bounds[0][2]])
print("итог: размер мм", out.extents.round(2), "замкнута:", out.is_watertight,
      "объём смолы %.2f см³ (сплошная была %.2f)" % (out.volume / 1000, solid.sum() * PITCH**3 / 1000))
out.export(dst)
