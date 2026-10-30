"""Двойной криптекс: два кода, два тайника.

Код A (WORD_A) -> слева выезжает весь тубус A (основной тайник).
Код B (WORD_B) -> справа снимается торец вместе с потайной капсулой-свитком B.

Ось вдоль X. Угол a — поворот вокруг X от +Z (верх); a=90 — передняя
сторона (-Y), там линия чтения. Зубья тубуса A сверху (a=0), зубья капсулы B
снизу (a=180). В каждом кольце два паза: узкий-глубокий (для A) и
широкий-мелкий (для B), поэтому зуб A не проходит в паз B и наоборот.

Запуск:  python cryptex.py      (поменять слова — WORD_A / WORD_B ниже)
"""
from __future__ import annotations
import math
from pathlib import Path
from cadgen import build123d as bd

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

WORD_A = "SOFIA"   # открывает слева тубус
WORD_B = "APPLE"   # открывает справа потайной свиток
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N = len(ALPHABET)
STEP_DEG = 360.0 / N

# ---- зазоры (смола; для FDM поднять CLR до 0.3, CLR_T до 0.3)
CLR = 0.2      # радиальный зазор скольжения
CLR_T = 0.25   # зазор зуб/паз на сторону

# ---- радиусы
R_SLV_IN, R_SLV_OUT = 16.0, 17.8      # гильза (статор)
R_CORE = R_SLV_IN - CLR               # тубус A
R_BORE = R_SLV_OUT + CLR              # отверстие кольца
R_OUT = 25.0                           # кольца и торцы, Ø50
H_A, H_B = 2.6, 1.05                   # высота зубьев над гильзой / в кольце
R_TAB_A = R_SLV_OUT + H_A              # 20.4
R_HEAD_B0, R_HEAD_B1 = R_SLV_OUT + 0.15, R_SLV_OUT + 0.15 + H_B  # 17.95..19.0
R_GROOVE = R_TAB_A + 0.4               # кольцевая канавка внутри кольца
R_NOTCH_B = R_HEAD_B1 + 0.35           # мелкий паз B (зуб A выше -> не пройдёт)
W_A, W_B, W_STEM = 3.4, 7.5, 3.4       # ширина зуба A, головки B, ножки B
W_SLOT = W_STEM + 2 * CLR_T

# капсула B и тайник A
R_CAP_B = 4.5
Z_CAP_B = -(R_CORE - R_CAP_B)
R_CAP_B_IN = 3.4                       # свиток Ø6.8
R_CARGO, Z_CARGO = 8.0, 3.3            # тайник A Ø16

# ---- по длине
RINGS = len(WORD_A)
RW, GW, GAP, TL = 14.0, 8.0, 0.25, 4.0  # кольцо, канавка, зазор, длина зуба
FL, COLLAR, CAP_T = 5.0, 5.0, 6.0
X_RING0 = FL + GAP
PITCH = RW + GAP
X_COLLAR = X_RING0 + RINGS * PITCH - GAP + GAP
X_END = X_COLLAR + COLLAR              # конец гильзы
GROOVE_X = [X_RING0 + i * PITCH + RW / 2 for i in range(RINGS)]
LONG = 200.0


def cyl(r, x0, x1, y=0.0, z=0.0):
    return bd.Pos(x0, y, z) * bd.Rot(0, 90, 0) * bd.Cylinder(
        r, x1 - x0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


def tube(r0, r1, x0, x1):
    return cyl(r1, x0, x1) - cyl(r0, x0 - 1, x1 + 1)


def rad_box(x0, x1, w, r0, r1, a_deg=0.0):
    """Брусок по X [x0,x1], ширина w по Y, по радиусу r0..r1 в сторону +Z, затем поворот на a."""
    b = bd.Pos((x0 + x1) / 2, 0, (r0 + r1) / 2) * bd.Box(x1 - x0, w, r1 - r0)
    return bd.Rot(a_deg, 0, 0) * b


def ring_rot(a_deg):
    return bd.Rot(a_deg, 0, 0)


def chamfer_edges(shape, d):
    try:
        return bd.chamfer(shape.edges().filter_by(bd.GeomType.CIRCLE), d)
    except Exception:
        return shape


# ---------------------------------------------------------------- кольцо
def letter_cut(ch, x_c):
    t = bd.Text(ch, 7.0, font_path=FONT, align=(bd.Align.CENTER, bd.Align.CENTER))
    # текст в плоскости XZ, смотрит на -Y, буква «вверх» = +Z
    sk = bd.Plane.XZ * t
    solid = bd.extrude(sk, amount=3.0)  # XZ-нормаль = -Y -> выдавливает в -Y
    return bd.Pos(x_c, -(R_OUT - 0.6), 0) * solid


def ring(i, letters=True):
    a_ch, b_ch = ALPHABET.index(WORD_A[i]), ALPHABET.index(WORD_B[i])
    n1 = a_ch * STEP_DEG - 90.0          # паз A в системе кольца
    n2 = b_ch * STEP_DEG + 90.0          # паз B
    x0 = 0.0
    body = tube(R_BORE, R_OUT, x0, RW)
    body -= tube(R_BORE - 1, R_GROOVE, (RW - GW) / 2, (RW + GW) / 2)
    body -= rad_box(-1, RW + 1, W_A + 2 * CLR_T, R_BORE - 1, R_GROOVE, n1)
    body -= rad_box(-1, RW + 1, W_B + 2 * CLR_T, R_BORE - 1, R_NOTCH_B, n2)
    # фаски по наружным кромкам
    body -= tube(R_OUT - 0.7, R_OUT + 2, -1, 0.0) + tube(R_OUT - 0.7, R_OUT + 2, RW, RW + 1)
    for k, ch in enumerate(ALPHABET if letters else ""):
        body -= ring_rot(k * STEP_DEG - 90.0) * letter_cut(ch, RW / 2)
    # риски-разделители между буквами
    for k in range(N if letters else 0):
        body -= ring_rot((k + 0.5) * STEP_DEG) * rad_box(1.2, RW - 1.2, 0.6, R_OUT - 0.4, R_OUT + 1)
    body.label = f"ring_{i+1}"
    return body


def ring_placed(i, code, letters=True):
    """Кольцо на своём месте, повёрнуто так, что на линии чтения буква code[i]."""
    k = ALPHABET.index(code[i])
    return bd.Pos(X_RING0 + i * PITCH, 0, 0) * ring_rot(90.0 - k * STEP_DEG) * ring(i, letters)


# ---------------------------------------------------------------- гильза + левый фланец
def sleeve():
    s = tube(R_SLV_IN, R_SLV_OUT, 0, X_END) + tube(R_SLV_IN, R_OUT, 0, FL)
    s -= tube(R_OUT - 0.7, R_OUT + 2, -1, 0.0) + tube(R_OUT - 0.7, R_OUT + 2, FL, FL + 1)
    # паз A сверху: открыт слева, до последнего зуба
    s -= rad_box(-1, GROOVE_X[-1] + TL / 2 + 1.5, W_SLOT, R_SLV_IN - 1, R_SLV_OUT + 1)
    # ворота A во фланце (под высокий зуб)
    s -= rad_box(-1, FL + 1, W_A + 2 * CLR_T, R_SLV_IN - 1, R_GROOVE)
    # паз B снизу: открыт справа, от первого зуба
    s -= rad_box(GROOVE_X[0] - TL / 2 - 1.5, X_END + 1, W_SLOT, R_SLV_IN - 1, R_SLV_OUT + 1, 180)
    # риска линии чтения спереди
    s -= rad_box(-1, FL + 1, 1.0, R_OUT - 0.6, R_OUT + 1, 90)
    s.label = "sleeve"
    return s


def collar():
    c = tube(R_SLV_OUT + 0.05, R_OUT, X_COLLAR, X_END)
    c -= tube(R_OUT - 0.7, R_OUT + 2, X_COLLAR - 1, X_COLLAR) + tube(R_OUT - 0.7, R_OUT + 2, X_END, X_END + 1)
    c -= rad_box(X_COLLAR - 1, X_END + 1, W_B + 2 * CLR_T, R_SLV_OUT - 1, R_NOTCH_B, 180)
    c -= rad_box(X_COLLAR - 1, X_END + 1, 1.0, R_OUT - 0.6, R_OUT + 1, 90)
    c.label = "collar"
    return c


def end_cap(x0, x1, outer_face_x):
    """Одинаковый декоративный торец для обеих сторон."""
    c = cyl(R_OUT, x0, x1)
    c = bd.chamfer(c.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.X)[0 if outer_face_x < x1 else -1], 1.2)
    sgn = -1 if outer_face_x < x1 else 1
    for r in (19.0, 12.0):
        groove = tube(r - 0.5, r + 0.5, outer_face_x - sgn * 0.8 - 0.0001 if sgn > 0 else outer_face_x, outer_face_x + 1 if sgn > 0 else outer_face_x + 0.8)
        c -= groove
    # насечка по ободу
    for k in range(36):
        c -= ring_rot(k * 10) * rad_box(x0 + 1.5, x1 - 1.5, 1.2, R_OUT - 0.5, R_OUT + 1)
    return c


# ---------------------------------------------------------------- тубус A
def core_a():
    x1 = X_END - 0.3
    c = cyl(R_CORE, 0, x1) + end_cap(-CAP_T, 0, -CAP_T)
    c -= cyl(R_CARGO, 2.0, x1 + 1, z=Z_CARGO)                     # тайник, открыт справа
    c -= cyl(R_CAP_B + CLR + 0.05, 0, x1 + 1, z=Z_CAP_B)          # канал под капсулу B
    c -= rad_box(0, x1 + 1, W_SLOT, R_CORE - 3.5, R_CORE + 1, 180)  # щель под ножку B
    for gx in GROOVE_X:
        c += rad_box(gx - TL / 2, gx + TL / 2, W_A, R_CORE - 1.5, R_TAB_A)
    c.label = "core_A"
    return c


# ---------------------------------------------------------------- капсула B (свиток) + правый торец
def capsule_b():
    xc0, xc1 = 0.5, X_END + 0.2
    cap = end_cap(xc1, xc1 + CAP_T, xc1 + CAP_T)
    k = cyl(R_CAP_B, xc0, xc1 + 0.5, z=Z_CAP_B) - cyl(R_CAP_B_IN, xc0 - 1, xc1 - 1.5, z=Z_CAP_B)
    # ножка вдоль всей длины в пазу гильзы
    fin = rad_box(GROOVE_X[0] - TL / 2, xc1 + 0.5, W_STEM, -Z_CAP_B, R_SLV_OUT - 0.1, 180)
    b = cap + k + fin
    for gx in GROOVE_X:
        b += rad_box(gx - TL / 2, gx + TL / 2, W_STEM, R_SLV_OUT - 0.5, R_HEAD_B1, 180)
        b += rad_box(gx - TL / 2, gx + TL / 2, W_B, R_HEAD_B0, R_HEAD_B1, 180)
    b.label = "capsule_B"
    return b


PARTS = {
    "sleeve": sleeve, "collar": collar, "core_A": core_a, "capsule_B": capsule_b,
}


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, fn in PARTS.items():
        shapes[name] = fn()
    for i in range(RINGS):
        shapes[f"ring_{i+1}"] = ring(i)
    for name, s in shapes.items():
        print(name, "volume", round(s.volume, 1), "valid", s.is_valid)
        bd.export_stl(s, str(out / f"{name}.stl"), tolerance=0.02, angular_tolerance=0.15)
    return shapes


if __name__ == "__main__":
    build(HERE)
