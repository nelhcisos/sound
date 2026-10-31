"""Раскладка готовых STL по столу Elegoo Saturn 3/4 Ultra (218,88 × 122,88 мм).
    python make_plates.py   (нужен trimesh)"""
from pathlib import Path
import trimesh

HERE = Path(__file__).resolve().parents[1]
PLATE_W, PLATE_H = 218.88, 122.88
cols = [27.4, 82.1, 136.8, 191.5]        # 4 × 2 ячейки по Ø50
rows = [30.7, 92.1]


def load(folder, name):
    return trimesh.load(folder / f"{name}.stl")


def plate(folder, cells, out):
    meshes = []
    for (c, r), names in cells.items():
        for n in names:                      # несколько деталей в ячейке — вложены (капсула внутри упора)
            m = load(folder, n)
            m.apply_translation([cols[c] - PLATE_W / 2, rows[r] - PLATE_H / 2, 0])
            meshes.append(m)
    full = trimesh.util.concatenate(meshes)
    lo, hi = full.bounds
    assert lo[0] >= -PLATE_W / 2 and hi[0] <= PLATE_W / 2 and lo[1] >= -PLATE_H / 2 and hi[1] <= PLATE_H / 2
    full.export(out)
    print(out.name, f"деталей {len(meshes)}, высота {hi[2]:.1f} мм, смолы {sum(m.volume for m in meshes)/1000:.0f} мл")


full = HERE
plate(full, {(0, 0): ["sleeve"], (1, 0): ["core_A_tube"], (2, 0): ["ring_1"], (3, 0): ["ring_2"],
             (0, 1): ["collar", "capsule_B_tube"], (1, 1): ["ring_3"], (2, 1): ["ring_4"], (3, 1): ["ring_5"]},
      HERE / "plates" / "saturn_plate1_main.stl")
plate(full, {(1, 0): ["cap_A"], (2, 0): ["cap_B"]}, HERE / "plates" / "saturn_plate2_caps.stl")
test = HERE.parent / "cryptex_test"
plate(test, {(0, 0): ["sleeve"], (1, 0): ["core_A_tube"], (2, 0): ["ring_1"], (3, 0): ["collar", "capsule_B_tube"],
             (1, 1): ["cap_A"], (2, 1): ["cap_B"]}, test / "plate" / "saturn_plate_test.stl")
