import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cryptex5 as C
from cadgen import build123d as bd
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cx = C.Cryptex5()
fr, pl = cx.frame(), cx.end_plate()
dr = [cx.drawer(k) for k in range(5)]
rg = [cx.ring_placed(i, cx.words[0], cx.ring(i)) for i in range(cx.n)]
def scene(pull):
    ch = [fr, pl] + rg + [bd.Pos(C.EXIT[k] * pull[k], 0, 0) * dr[k] for k in range(5)]
    return bd.Compound(children=ch)
bd.export_step(scene([0] * 5), str(out / "closed.step"))
bd.export_step(scene([70, 0, 0, 0, 0]), str(out / "open1.step"))
bd.export_step(scene([70, 45, 20, 45, 70]), str(out / "open_all.step"))
