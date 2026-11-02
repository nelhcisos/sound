"""Раскладка под Saturn 3/4 Ultra (218,88 × 122,88 мм). Нужны trimesh, shapely."""
from pathlib import Path
import trimesh
from shapely.geometry import MultiPoint
H = Path(__file__).resolve().parents[1]
PW, PH = 218.88, 122.88
def plate(items, out):
    ms, foot = [], []
    for name, x, y in items:
        m = trimesh.load(H / f"{name}.stl"); m.apply_translation([x - PW / 2, y - PH / 2, 0]); ms.append(m)
        foot.append((name, MultiPoint(m.vertices[:, :2]).convex_hull))
    for i in range(len(foot)):
        for j in range(i + 1, len(foot)):
            assert foot[i][1].distance(foot[j][1]) > 1.0, (foot[i][0], foot[j][0])
    a = trimesh.util.concatenate(ms); lo, hi = a.bounds
    assert lo[0] >= -PW / 2 and hi[0] <= PW / 2 and lo[1] >= -PH / 2 and hi[1] <= PH / 2, (lo, hi)
    a.export(out); print(out.name, len(ms), "деталей, высота", round(hi[2], 1), "мм,", round(sum(m.volume for m in ms) / 1000), "мл")
plate([("frame", 30, 29), ("ring_1", 109.4, 29), ("ring_2", 188.8, 29), ("ring_3", 188.8, 93),
       ("drawer_1", 17, 104), ("drawer_2", 45, 104), ("drawer_3", 73, 104), ("drawer_4", 101, 104), ("drawer_5", 129, 104)],
      H / "plates" / "saturn_plate1_main.stl")
plate([("ring_4", 30, 29), ("ring_5", 109.4, 29), ("end_plate", 188.8, 29)], H / "plates" / "saturn_plate2.stl")
