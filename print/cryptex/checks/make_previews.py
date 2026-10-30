"""Сцены для превью: закрыт (SOFIA), открыт A, открыт B, разборка."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cryptex as C
from cadgen import build123d as bd

out = HERE / "views"; out.mkdir(exist_ok=True)
core, cap, slv, col = C.core_a(), C.capsule_b(), C.sleeve(), C.collar()

def lab(s, n):
    s.label = n; return s

def scene(code, dx_a=0.0, dx_b=0.0, dz=0.0):
    parts = [lab(slv, "sleeve"), lab(col, "collar"),
             lab(bd.Pos(dx_a, 0, 0) * core, "core_A"), lab(bd.Pos(dx_b, 0, 0) * cap, "capsule_B")]
    parts += [lab(C.ring_placed(i, code), f"ring_{i+1}") for i in range(C.RINGS)]
    return bd.Pos(0, 0, dz) * bd.Compound(children=parts)

bd.export_step(scene(C.WORD_A), str(out / "closed.step"))
bd.export_step(bd.Compound(children=[
    scene(C.WORD_A, dx_a=-70),
    bd.Pos(0, 0, -70) * scene(C.WORD_B, dx_b=70)]), str(out / "open_both.step"))
