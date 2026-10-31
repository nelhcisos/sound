"""STEP-сцены для превью: закрыт (SOFIA), открыт кодом A и кодом B."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cryptex as C
from cadgen import build123d as bd

out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "views"
out.mkdir(parents=True, exist_ok=True)
cx = C.Cryptex()
core, caps, slv, col = cx.core_assembled(), cx.capsule_assembled(), cx.sleeve(), cx.collar()
rings = [cx.ring(i) for i in range(cx.n)]


def lab(s, n):
    s.label = n
    return s


def scene(code, dx_a=0.0, dx_b=0.0):
    parts = [lab(slv, "sleeve"), lab(col, "collar"),
             lab(bd.Pos(dx_a, 0, 0) * core, "core_A"), lab(bd.Pos(dx_b, 0, 0) * caps, "capsule_B")]
    parts += [lab(cx.ring_placed(i, code, ring_shape=rings[i]), f"ring_{i + 1}") for i in range(cx.n)]
    return bd.Compound(children=parts)


bd.export_step(scene(cx.wa), str(out / "closed.step"))
bd.export_step(bd.Compound(children=[scene(cx.wa, dx_a=-72), bd.Pos(0, 0, -70) * scene(cx.wb, dx_b=72)]),
               str(out / "open_both.step"))
