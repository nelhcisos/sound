"""Криптекс-подставка для фигурки «Хирург» (cadgen / build123d).
Вертикальный: сверху торец A — подставка фигурки (код SOFIA -> тянешь фигурку, выезжает тубус),
снизу торец B — потайной свиток (код APPLE).
    python build_surgeon.py      -> STL деталей криптекса в эту папку (торец A пока без гнёзд)
    затем: python surgeon_sockets.py source/base_original.stl (trimesh + manifold3d) -> гнёзда под кроксы, фигурка, превью
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cryptex"))
import cryptex as C
from cadgen import build123d as bd

cx = C.Cryptex(figure=True)
total = 0.0
for name, s in cx.print_parts().items():
    bb = s.bounding_box()
    print(f"{name:16s} {s.volume / 1000:6.1f} мл  {bb.size.X:5.1f}×{bb.size.Y:5.1f}×{bb.size.Z:5.1f} мм valid={s.is_valid}")
    total += s.volume
    out = HERE / ("_cap_A_blank.stl" if name == "cap_A" else f"{name}.stl")
    bd.export_stl(s, str(out), tolerance=0.01, angular_tolerance=0.1)
print(f"итого ~{total / 1000:.0f} мл")

# сборка (система криптекса) для превью: закрыт на SOFIA и открыт на SOFIA
core, caps, slv, col = cx.core_assembled(), cx.capsule_assembled(), cx.sleeve(), cx.collar()
rings = [cx.ring_placed(i, cx.wa) for i in range(cx.n)]
for tag, dx in (("closed", 0.0), ("openA", -70.0)):
    scene = [slv, col, bd.Pos(dx, 0, 0) * core, caps] + rings
    bd.export_stl(bd.Compound(children=scene), str(HERE / f"_scene_{tag}.stl"), tolerance=0.03, angular_tolerance=0.2)
print("x_cap_top", -C.CAP_T_FIG, "R_OUT", C.R_OUT)
