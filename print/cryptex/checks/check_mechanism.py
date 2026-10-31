"""Проверка механики: пересечения в сборе, при сдвиге колец и при выдвижении A/B для разных кодов."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cryptex as C
from cadgen import build123d as bd

words = sys.argv[1:3] if len(sys.argv) > 2 else (C.WORD_A, C.WORD_B)
cx = C.Cryptex(*words)
tube_a, cap_a, tube_b, cap_b = cx.core_tube(), cx.cap_a(), cx.capsule_tube(), cx.cap_b()
core, caps = tube_a + cap_a, tube_b + cap_b
slv, col = cx.sleeve(), cx.collar()
rings_raw = [cx.ring(i, letters=False) for i in range(cx.n)]
code_a, code_b = cx.wa, cx.wb


def vol(a, b):
    x = a & b
    return x.volume if x else 0.0


def rings(code, shift=None):
    out = []
    for i, r in enumerate(rings_raw):
        dx = shift[i] if shift else 0.0
        out.append(bd.Pos(dx, 0, 0) * cx.ring_placed(i, code, ring_shape=r))
    return out


def inter(a, others):
    return sum(vol(a, o) for o in others)


print("Клеевые посадки (бобышка торца не должна задевать трубку):",
      round(vol(cap_a, tube_a), 4), round(vol(cap_b, tube_b), 4))

n = cx.n
push_right = [C.GAP * (n - i) for i in range(n)]
push_left = [-C.GAP * (i + 1) for i in range(n)]
print("\nВ сборе, пересечения, мм³ (должно быть 0):")
for code in (code_a, code_b, "X" * n, "M" * n):
    for label, sh in (("кольца по центру", None), ("сдвинуты вправо", push_right), ("сдвинуты влево", push_left)):
        R = rings(code, sh)
        v = (inter(core, R + [slv, col, caps]) + inter(caps, R + [slv, col])
             + inter(slv, R + [col]) + sum(vol(R[i], R[i + 1]) for i in range(n - 1)) + vol(R[-1], col))
        print(f"  {code} {label:16s} {v:.3f}")


def sweep(part, others, direction, dist, step=1.0):
    d, worst = 0.0, 0.0
    while d <= dist:
        v = inter(bd.Pos(direction * d, 0, 0) * part, others)
        worst = max(worst, v)
        if v > 1.0:
            return f"ЗАБЛОКИРОВАН на {d:.0f} мм (пересечение {v:.1f} мм³)"
        d += step
    return f"свободно выходит на {dist:.0f} мм (макс. пересечение {worst:.3f} мм³)"


A = C.ALPHABET
wrong1 = code_a[:-1] + A[(A.index(code_a[-1]) + 1) % 26]
wrong2 = code_b[:-1] + A[(A.index(code_b[-1]) - 1) % 26]
cross = "".join(A[(A.index(code_b[i]) + 13) % 26] for i in range(n))   # паз B под зубом A
cross2 = "".join(A[(A.index(code_a[i]) + 13) % 26] for i in range(n))  # паз A под зубом B
dist = cx.x_end + 8
for code, label in [(code_a, "код A"), (code_b, "код B"), (wrong1, "A с ошибкой в 1 букве"),
                    (wrong2, "B с ошибкой в 1 букве"), (cross, "паз B подставлен под зуб A"),
                    (cross2, "паз A подставлен под зуб B")]:
    R = rings(code)
    print(f"\n{label} = {code}")
    print("  тубус A влево:   ", sweep(core, R + [slv, col, caps], -1, dist))
    print("  капсула B вправо:", sweep(caps, R + [slv, col, core], +1, dist))
