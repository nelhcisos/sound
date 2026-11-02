"""Механика: в сборе без пересечений (и при сдвиге колец); каждое слово выпускает только свою секцию."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cryptex5 as C
from cadgen import build123d as bd

cx = C.Cryptex5()
frame, plate = cx.frame(), cx.end_plate()
drawers = [cx.drawer(k) for k in range(5)]
rr = [cx.ring(i, letters=False) for i in range(cx.n)]
n = cx.n


def vol(a, b):
    x = a & b
    return x.volume if x else 0.0


def rings(code, shift=None):
    return [bd.Pos(shift[i] if shift else 0, 0, 0) * cx.ring_placed(i, code, rr[i]) for i in range(n)]


push_r = [C.GAP * (n - i) for i in range(n)]
push_l = [-C.GAP * (i + 1) for i in range(n)]
print("В сборе (мм³, должно быть 0):")
for code in cx.words + ["XXXXX"]:
    for lab, sh in (("центр", None), ("вправо", push_r), ("влево", push_l)):
        R = rings(code, sh)
        v = sum(vol(d, o) for d in drawers for o in R + [frame, plate])
        v += sum(vol(drawers[i], drawers[j]) for i in range(5) for j in range(i + 1, 5))
        v += sum(vol(frame, o) for o in R + [plate]) + sum(vol(R[i], R[i + 1]) for i in range(n - 1)) + vol(R[-1], plate)
        print(f"  {code} {lab:6s} {v:.3f}", flush=True)


def sweep(k, R, dist=100, step=2.0):
    d, others = 0.0, R + [frame, plate] + [drawers[j] for j in range(5) if j != k]
    while d <= dist:
        v = sum(vol(bd.Pos(C.EXIT[k] * d, 0, 0) * drawers[k], o) for o in others)
        if v > 1.0:
            return f"стоит ({d:.0f} мм)"
        d += step
    return "ВЫЕЗЖАЕТ"


A = C.ALPHABET
for k, w in enumerate(cx.words):
    for code, lab in ((w, "слово"), (w[:-1] + A[(A.index(w[-1]) + 1) % 26], "ошибка в 1 букве")):
        R = rings(code)
        res = [sweep(j, R) for j in range(5)]
        print(f"{lab:17s} {code}: " + ", ".join(f"секция {j + 1}: {r}" for j, r in enumerate(res)), flush=True)
