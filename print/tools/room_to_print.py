"""Комната из 3D Jutsu (GLB) -> один цельный STL для печати смолой.

python3 print/tools/room_to_print.py <room.glb> <out.stl> [масштаб, по умолч. 50]

- выкидывает потолок, люстру, южную стену и нишу с входной дверью (вид «кукольный домик»);
- все детали чуть раздуваются (0,1 мм на печати), чтобы касающиеся слились;
- тонкое доводится до минимума 0,5 мм на печати;
- под полом основание 3 мм;
- всё объединяется в одно замкнутое тело (manifold3d), Z вверх, мм.
"""
import sys
import numpy as np
import trimesh
import manifold3d as mf

src, out = sys.argv[1], sys.argv[2]
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
K = 1000.0 / SCALE           # метры -> мм печати
MIN_MM = 0.5                  # минимальная толщина на печати
GROW_MM = 0.1                 # раздувание для слияния касающихся деталей
BASE_MM = 3.0

SKIP = ("ENV_ceiling", "ENV_wall_south", "ENV_wall_recess", "ENV_baseboard_south",
        "ENV_cornice_south", "PRP_entry_", "PRP_ceiling_lamp", "PRP_rug_line")

# доп. утолщение (мм, X/Y/Z) — чтобы деталь дотянулась до стены
EXTRA = {"PRP_radiator_fin": (0, 1.6, 0)}
# объёмные «чаши» — печатать сплошными (иначе замкнутая полость со смолой)
SOLIDIFY = ("FUR_chair_shell",)

scene = trimesh.load(src)
# glTF: Y вверх -> Z вверх
yup = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

parts = []
RNG = np.random.default_rng(7)
for node in scene.graph.nodes_geometry:
    if node.startswith(SKIP):
        continue
    T, g = scene.graph[node]
    m = scene.geometry[g].copy()
    m.apply_transform(T)
    m.apply_transform(yup)
    m.apply_scale(K)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    if not m.is_watertight or node.startswith(SOLIDIFY):
        m = m.convex_hull
        print("convex hull:", node)
    # толщина и раздувание по осям вокруг центра bbox
    lo, hi = m.bounds
    ext = hi - lo
    c = (lo + hi) / 2
    # небольшой разброс, чтобы грани соседних деталей не совпадали ровно (иначе рёбра «на 4 грани»)
    target = np.maximum(ext, MIN_MM) + 2 * GROW_MM + RNG.uniform(0.0, 0.03, 3)
    for pre, extra in EXTRA.items():
        if node.startswith(pre):
            target = target + np.array(extra)
    s = target / np.maximum(ext, 1e-6)
    S = np.eye(4); S[:3, :3] = np.diag(s)
    m.apply_transform(trimesh.transformations.translation_matrix(c) @ S @
                      trimesh.transformations.translation_matrix(-c))
    if m.volume < 0:
        m.invert()
    parts.append((node, m))

def to_mf(m):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))

solids = []
for n, m in parts:
    x = to_mf(m)
    if x.status() != mf.Error.NoError or x.is_empty():
        print("skip broken:", n, x.status()); continue
    solids.append(x)

# комната без южной ниши: x -0.1..3.5, y 0..4.9 м
x0, x1, y0, y1 = -0.1 * K, 3.5 * K, -0.1 * K, 4.9 * K
base = mf.Manifold.cube([x1 - x0, y1 - y0, BASE_MM]).translate([x0, y0, -BASE_MM])
solids.append(base)
u = mf.Manifold.batch_boolean(solids, mf.OpType.Add)
clip = mf.Manifold.cube([x1 - x0, y1 - y0, 200]).translate([x0, y0, -BASE_MM])
u = u ^ clip

# полости внутри (отрицательный объём) заливаем, отдельные куски показываем
pieces = u.decompose()
keep = [p for p in pieces if p.volume() > 0]
voids = len(pieces) - len(keep)
u = mf.Manifold.batch_boolean(keep, mf.OpType.Add) if len(keep) > 1 else keep[0]
print("кусков:", len(keep), "залито полостей:", voids)
for p in sorted(keep, key=lambda p: p.volume())[:-1]:
    print("  отдельный кусок, bbox:", p.bounding_box(), "V=%.2f" % p.volume())
u = u.simplify(0.01)   # убрать микро-щепки (<0,01 мм)
mesh = u.to_mesh()
res = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)
res.apply_translation(-res.bounds[0])          # в угол, дно на Z=0
print("объём см3: %.1f" % (res.volume / 1000), "размер мм:", np.round(res.extents, 1),
      "watertight:", res.is_watertight)
res.export(out)
