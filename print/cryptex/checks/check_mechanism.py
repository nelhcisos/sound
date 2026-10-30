"""Проверка механики: пересечения в собранном виде и при выдвижении A/B для разных кодов."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cryptex as C
from cadgen import build123d as bd

core, cap, slv, col = C.core_a(), C.capsule_b(), C.sleeve(), C.collar()
rings_raw = [C.ring(i, letters=False) for i in range(C.RINGS)]

def rings(code):
    out = []
    for i, r in enumerate(rings_raw):
        k = C.ALPHABET.index(code[i])
        out.append(bd.Pos(C.X_RING0 + i * C.PITCH, 0, 0) * C.ring_rot(90.0 - k * C.STEP_DEG) * r)
    return out

def inter(a, others):
    v = 0.0
    for o in others:
        x = a & o
        v += x.volume if x else 0.0
    return v

def sweep(part, others, direction, dist=95, step=2.0):
    worst = 0.0; at = None
    d = 0.0
    while d <= dist:
        v = inter(bd.Pos(direction * d, 0, 0) * part, others)
        if v > worst: worst, at = v, d
        if v > 1.0: return f"ЗАБЛОКИРОВАН на {d:.0f} мм (пересечение {v:.1f} мм³)"
        d += step
    return f"свободно выходит на {dist} мм (макс. пересечение {worst:.3f} мм³)"

static = [slv, col]
code_a, code_b = C.WORD_A, C.WORD_B
print("Собранно, пересечения (должно быть 0):")
for code in (code_a, code_b, "XXXXX"):
    R = rings(code)
    print(" ", code, "A:", round(inter(core, R + static + [cap]), 3), "B:", round(inter(cap, R + static), 3),
          "кольца-гильза:", round(inter(slv, R + [col]), 3))

wrong1 = code_a[:2] + C.ALPHABET[(C.ALPHABET.index(code_a[2]) + 1) % 26] + code_a[3:]
# «перекрёстный» код: паз B стоит сверху (там, где зуб A)
cross = "".join(C.ALPHABET[(C.ALPHABET.index(code_b[i]) + 13) % 26] for i in range(C.RINGS))
cross2 = "".join(C.ALPHABET[(C.ALPHABET.index(code_a[i]) + 13) % 26] for i in range(C.RINGS))
for code, label in [(code_a, "код A"), (code_b, "код B"), (wrong1, "A с ошибкой в 1 букве"),
                    (cross, "паз B подставлен под зуб A"), (cross2, "паз A подставлен под зуб B")]:
    R = rings(code)
    print(f"\n{label} = {code}")
    print("  тубус A влево:   ", sweep(core, R + static, -1))
    print("  капсула B вправо:", sweep(cap, R + static, +1))
