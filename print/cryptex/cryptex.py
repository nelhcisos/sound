"""Двойной криптекс под фотополимерную печать (смола): два кода — два тайника.

Код A (WORD_A) -> слева выезжает тубус A (основной тайник).
Код B (WORD_B) -> справа снимается торец вместе с потайной капсулой-свитком B.

Система координат сборки: ось вдоль X. Угол a — поворот вокруг X от +Z (верх);
a=90 — передняя сторона (-Y), там линия чтения. Зубья тубуса A сверху (a=0),
зубья капсулы B снизу (a=180). В каждом кольце два паза: узкий-глубокий
(для A) и широкий-мелкий (для B), поэтому зуб A не проходит в паз B и наоборот.

Под смолу: каждая деталь печатается ПРЯМО НА СТОЛЕ, БЕЗ ПОДДЕРЖЕК —
нависания сделаны скосами 45°, у плоскости стола фаски против «слоновьей ноги»,
полости с вентиляционными канавками (без эффекта присоски). Торцы — отдельные
детали на клею (иначе нужны длинные поддержки).

Запуск (cadgen / build123d):
    python cryptex.py            -> полный криптекс в эту папку
    python cryptex.py test       -> тест-набор на одно кольцо в ../cryptex_test
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from cadgen import build123d as bd

HERE = Path(__file__).resolve().parent
FONT = str(HERE / "fonts" / "DejaVuSerif-Bold.ttf")

WORD_A = "SOFIA"   # открывает слева тубус
WORD_B = "SDES "   # открывает справа потайной свиток (5-я позиция — пустая)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "   # 27-я позиция — пустая (без буквы)
N = len(ALPHABET)
STEP_DEG = 360.0 / N

# ---------------------------------------------------------------- зазоры (смола)
CLR = 0.20        # кольцо на гильзе, радиально
CLR_SLIDE = 0.25  # длинные скольжения: тубус в гильзе, капсула в канале
CLR_T = 0.25      # зуб/паз, ножка/прорезь — на сторону
CLR_GLUE = 0.15   # клеевые посадки (бобышки торцов) — на сторону
EF = 0.5          # фаска против «слоновьей ноги» у плоскости стола
LEAD = 0.6        # заходные фаски пазов (кольцо само доворачивается)

# ---------------------------------------------------------------- радиусы
R_SLV_IN, R_SLV_OUT = 16.0, 17.8      # гильза
R_CORE = R_SLV_IN - CLR_SLIDE         # тубус A
R_BORE = R_SLV_OUT + CLR              # отверстие кольца
R_OUT = 25.0                           # кольца и торцы, Ø50
R_TAB_A = 20.8                         # верх зуба A
R_GROOVE = 21.2                        # кольцевая канавка внутри кольца
R_HEAD_B0, R_HEAD_B1 = 18.1, 19.5      # головка зуба B
R_NOTCH_B = 19.85                      # мелкий паз B (зуб A выше -> не пройдёт)
R_LEDGE_CH = 19.6                      # выше — скос 45° уступа (печать без поддержек)
R_FIN_TOP = R_SLV_OUT - 0.1            # ножка капсулы в прорези гильзы
W_A, W_B, W_STEM = 3.4, 7.5, 3.4
W_NOTCH_A = W_A + 2 * CLR_T
W_NOTCH_B = W_B + 2 * CLR_T
W_SLOT = W_STEM + 2 * CLR_T

# капсула B, канал под неё в тубусе, тайник A
R_CAPS = 4.5
Z_CAPS = -(R_CORE - R_CAPS)
R_CAPS_IN = 3.4                        # свиток Ø6,8
R_CH = R_CAPS + CLR_SLIDE
R_CARGO = 13.4                         # тайник A ~Ø26
WALL_CH = 1.6                          # стенка между тайником и каналом

# ---------------------------------------------------------------- по длине
RW, GW, GAP = 14.0, 8.6, 0.2           # кольцо, канавка, зазор между кольцами
FL, COLLAR_W, CAP_T, BOSS_H = 5.0, 5.0, 6.0, 3.0
CAP_T_FIG = 7.0                        # торец-подставка фигурки
PIN_R, PIN_L, PIN_RAD = 0.8, 2.5, 14.6  # штырьки тубус -> торец-подставка
PIN_ANGLES = (60.0, -110.0)             # несимметрично: торец встаёт только одним положением
TAB_A_X0, TAB_A_X1 = -3.0, 0.0         # верх зуба A относительно центра канавки
HEAD_B_X0, HEAD_B_X1 = -1.6, 2.4       # верх головки B


# ================================================================ примитивы
def rot(a):
    return bd.Rot(a, 0, 0)


def cyl(r, x0, x1, y=0.0, z=0.0):
    return bd.Pos(x0, y, z) * bd.Rot(0, 90, 0) * bd.Cylinder(
        r, x1 - x0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


def tube(r0, r1, x0, x1):
    return cyl(r1, x0, x1) - cyl(r0, x0 - 1, x1 + 1)


def rad_box(x0, x1, w, r0, r1, a=0.0):
    """Брусок: X [x0,x1], ширина w по Y, по радиусу r0..r1 вверх (+Z), повёрнут на a."""
    return rot(a) * (bd.Pos((x0 + x1) / 2, 0, (r0 + r1) / 2) * bd.Box(x1 - x0, w, r1 - r0))


def xr_prism(pts, w, a=0.0):
    """Призма из профиля (x, r) в плоскости XZ, толщиной w по касательной, повёрнута на a."""
    face = bd.Plane.XZ * bd.Polygon(*pts, align=None)
    return rot(a) * bd.extrude(face, amount=w / 2, both=True)


def revolve_xr(pts, y0=0.0, z0=0.0):
    """Тело вращения профиля (x, r) вокруг оси, параллельной X и проходящей через (y0, z0)."""
    return bd.Pos(0, y0, z0) * bd.revolve(bd.Polygon(*pts, align=None), axis=bd.Axis.X)


def edge_chamfer(x_face, r_edge, size, solid_dir, solid_out, y0=0.0, z0=0.0):
    """Срез 45° круговой кромки. solid_dir: материал в +x (1) или -x (-1) от торца;
    solid_out: материал снаружи радиуса (кромка отверстия) или внутри (наружная кромка)."""
    d, s = solid_dir, (1 if solid_out else -1)
    pts = [(x_face - d, r_edge - s), (x_face - d, r_edge + s * (size + 1)),
           (x_face + d * (size + 1), r_edge - s)]
    return revolve_xr(pts, y0, z0)


def vcut(x_face, d, w, c, r0, r1, a):
    """Заходная фаска паза шириной w на торце x_face (материал в сторону d), глубина c."""
    h = w / 2
    pts = [(x_face - d, -(h + c + 1)), (x_face, -(h + c)), (x_face + d * c, -h),
           (x_face + d * c, h), (x_face, h + c), (x_face - d, h + c + 1)]
    prism = bd.Pos(0, 0, r0) * bd.extrude(bd.Polygon(*pts, align=None), amount=r1 - r0)
    return rot(a) * prism


def gable(x_end, w, r0, r1, a=0.0):
    """Острый (45°) конец прорези — печатается без поддержки."""
    s = w / math.sqrt(2)
    return rot(a) * (bd.Pos(x_end, 0, (r0 + r1) / 2) * bd.Rot(0, 0, 45) * bd.Box(s, s, r1 - r0))


def plane_x(x0, sign=1):
    """Плоскость сечения YZ в точке x0, нормаль ±X (локальные u -> ±Y, v -> Z)."""
    if sign > 0:
        return bd.Plane.YZ.offset(x0)
    return bd.Plane(origin=(x0, 0, 0), x_dir=(0, -1, 0), z_dir=(-1, 0, 0))


def extrude_x(face2d, x0, x1, taper=0.0):
    sign = 1 if x1 >= x0 else -1
    return bd.extrude(plane_x(x0, sign) * face2d, amount=abs(x1 - x0), taper=taper)


# ================================================================ сечения (u=Y, v=Z)
def u_profile(grow=0.0):
    """Канал под капсулу B в тубусе A (U-образный, открыт вниз)."""
    r = R_CH + grow
    return bd.Pos(0, Z_CAPS) * bd.Circle(r) + bd.Pos(0, Z_CAPS - 10) * bd.Rectangle(2 * r, 20)


def cargo_profile(grow=0.0):
    """Тайник A: круг минус зона вокруг канала, углы скруглены."""
    raw = bd.Circle(R_CARGO) - bd.offset(u_profile(), WALL_CH)
    out = bd.offset(bd.offset(raw, -1.0), 1.0)
    return bd.offset(out, grow) if grow else out


def letter_cut(ch, x_c, upright=False):
    if upright:
        # вертикальный криптекс: буква стоит вдоль оси (верх буквы -> -X, к торцу A)
        t = bd.Text(ch, 8.0, font_path=FONT, align=(bd.Align.CENTER, bd.Align.CENTER))
        bb = t.bounding_box().size
        t = bd.scale(t, by=(min(1.0, 4.4 / bb.X), min(1.0, 7.0 / bb.Y), 1.0))
        t = bd.Rot(0, 0, 90) * t
    else:
        t = bd.Text(ch, 6.2, font_path=FONT, align=(bd.Align.CENTER, bd.Align.CENTER))
        h = t.bounding_box().size.Y
        if h > 4.6:                   # Q, J с хвостиками — ужать, чтобы не задевать риски
            t = bd.scale(t, 4.6 / h)
    solid = bd.extrude(bd.Plane.XZ * t, amount=3.0)       # Plane.XZ: нормаль -Y
    return bd.Pos(x_c, -(R_OUT - 0.6), 0) * solid


def ang_dist(a, b):
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


# ================================================================ криптекс
class Cryptex:
    def __init__(self, word_a=WORD_A, word_b=WORD_B, cap_t=CAP_T, figure=False):
        assert len(word_a) == len(word_b)
        self.wa, self.wb, self.n = word_a, word_b, len(word_a)
        self.cap_t = cap_t
        # figure=True: вертикальный криптекс, торец A — подставка фигурки (буквы вдоль оси,
        # торец A без бобышки, клеится на два штырька, чтобы фигурка встала лицом к коду)
        self.figure = figure
        self.dir = -1 if figure else 1   # у вертикального алфавит слева направо
        for a, b in zip(word_a, word_b):
            sep = ang_dist((ALPHABET.index(b) - ALPHABET.index(a)) * STEP_DEG + 180.0, 0.0)
            assert sep > 25.0, f"буквы {a}/{b}: пазы A и B наложатся — поменяй слово"
        self.x_ring0 = FL + GAP
        self.pitch = RW + GAP
        self.gx = [self.x_ring0 + i * self.pitch + RW / 2 for i in range(self.n)]
        self.x_collar = self.x_ring0 + self.n * self.pitch
        self.x_end = self.x_collar + COLLAR_W            # конец гильзы
        self.l_core = self.x_end - 0.3                   # длина трубки A
        self.xc0 = 0.5                                   # левый конец капсулы B
        self.xc1 = self.x_end + 0.2                      # внутренняя плоскость торца B
        self.x_fin0 = self.gx[0] + HEAD_B_X0 - (R_HEAD_B1 - R_FIN_TOP)

    # ------------------------------------------------------------ кольцо
    def notch_angles(self, i):
        a_ch, b_ch = ALPHABET.index(self.wa[i]), ALPHABET.index(self.wb[i])
        d = self.dir
        return d * a_ch * STEP_DEG - 90.0, d * b_ch * STEP_DEG + 90.0   # в системе кольца

    def vent_angles(self, n1, n2):
        best = None
        for o in range(N):
            angs = [((o + k) % N + 0.5) * STEP_DEG for k in (0, 9, 17)]
            score = min(min(ang_dist(a, n1), ang_dist(a, n2)) for a in angs)
            if best is None or score > best[0]:
                best = (score, angs)
        return best[1]

    def ring(self, i, letters=True):
        n1, n2 = self.notch_angles(i)
        xl, xr = (RW - GW) / 2, (RW + GW) / 2
        body = tube(R_BORE, R_OUT, 0, RW)
        # канавка: левый уступ прямой (держит зуб A). Правый при печати оказывается сверху:
        # от стенки он нарастает скосом 45° (R_GROOVE -> R_LEDGE_CH), ниже — короткий
        # плоский уступ 1,6 мм, в который упирается головка B.
        body -= revolve_xr([(xl, R_BORE - 1), (xl, R_GROOVE), (xr - (R_GROOVE - R_LEDGE_CH), R_GROOVE),
                            (xr, R_LEDGE_CH), (xr, R_BORE - 1)])
        body -= rad_box(-1, RW + 1, W_NOTCH_A, R_BORE - 1, R_GROOVE, n1)
        body -= rad_box(-1, RW + 1, W_NOTCH_B, R_BORE - 1, R_NOTCH_B, n2)
        body -= vcut(xl, -1, W_NOTCH_A, LEAD, R_BORE - 1, R_GROOVE, n1)     # заход A
        body -= vcut(xr, +1, W_NOTCH_B, LEAD, R_BORE - 1, R_NOTCH_B, n2)    # заход B
        # торец x=0 лежит на столе: фаски против «слоновьей ноги» + вентиляция
        body -= edge_chamfer(0, R_BORE, EF, +1, True)
        body -= vcut(0, +1, W_NOTCH_A, EF, R_BORE - 1, R_GROOVE, n1)
        body -= vcut(0, +1, W_NOTCH_B, EF, R_BORE - 1, R_NOTCH_B, n2)
        for a in self.vent_angles(n1, n2):
            body -= rad_box(-1, 0.6, 1.2, R_BORE - 1, R_OUT + 1, a)
        body -= edge_chamfer(0, R_OUT, 0.7, +1, False) + edge_chamfer(RW, R_OUT, 0.7, -1, False)
        # номер кольца на правом торце (после сборки не виден, при печати смотрит вверх)
        num = bd.Text(str(i + 1), 3.0, font_path=FONT, align=(bd.Align.CENTER, bd.Align.CENTER))
        face_r = bd.Plane(origin=(RW, 0, 0), x_dir=(0, -1, 0), z_dir=(-1, 0, 0))
        body -= bd.extrude(face_r * (bd.Pos(0, 23.0) * num), amount=0.4)
        if letters:
            for k, ch in enumerate(ALPHABET):
                if ch == " ":
                    continue
                body -= rot(self.dir * k * STEP_DEG - 90.0) * letter_cut(ch, RW / 2, self.figure)
            for k in range(N):
                body -= rot((k + 0.5) * STEP_DEG) * rad_box(1.2, RW - 1.2, 0.4, R_OUT - 0.4, R_OUT + 1)
        body.label = f"ring_{i + 1}"
        return body

    def ring_placed(self, i, code, letters=True, ring_shape=None):
        k = ALPHABET.index(code[i])
        r = ring_shape if ring_shape is not None else self.ring(i, letters)
        return bd.Pos(self.x_ring0 + i * self.pitch, 0, 0) * rot(90.0 - self.dir * k * STEP_DEG) * r

    # ------------------------------------------------------------ гильза + левый фланец
    def sleeve(self):
        s = tube(R_SLV_IN, R_SLV_OUT, 0, self.x_end) + tube(R_SLV_IN, R_OUT, 0, FL)
        s -= edge_chamfer(0, R_OUT, 0.7, +1, False) + edge_chamfer(FL, R_OUT, 0.7, -1, False)
        xa = self.gx[-1] + TAB_A_X1 + (R_TAB_A - R_SLV_IN) + 0.4        # прорезь A до последнего зуба
        s -= rad_box(-1, xa, W_SLOT, R_SLV_IN - 1, R_SLV_OUT + 1)
        s -= gable(xa, W_SLOT, R_SLV_IN - 1, R_SLV_OUT + 1)
        s -= rad_box(-1, FL + 1, W_NOTCH_A, R_SLV_IN - 1, R_GROOVE)        # ворота A во фланце
        xb = self.x_fin0 - (R_FIN_TOP - R_SLV_IN) - 0.4                    # прорезь B от первого зуба
        s -= rad_box(xb, self.x_end + 1, W_SLOT, R_SLV_IN - 1, R_SLV_OUT + 1, 180)
        s -= edge_chamfer(0, R_SLV_IN, EF, +1, True)                        # торец на столе
        s -= vcut(0, +1, W_NOTCH_A, EF, R_SLV_IN - 1, R_GROOVE, 0)
        s -= rad_box(-1, FL + 1, 1.0, R_OUT - 0.6, R_OUT + 1, 90)           # риска линии чтения
        s.label = "sleeve"
        return s

    def collar(self):
        x0, x1 = self.x_collar, self.x_end
        r_in = R_SLV_OUT + 0.1
        c = tube(r_in, R_OUT, x0, x1)
        c -= edge_chamfer(x0, R_OUT, 0.7, +1, False) + edge_chamfer(x1, R_OUT, 0.7, -1, False)
        c -= rad_box(x0 - 1, x1 + 1, W_NOTCH_B, R_SLV_OUT - 1, R_NOTCH_B, 180)
        c -= rad_box(x0 - 1, x1 + 1, 1.0, R_OUT - 0.6, R_OUT + 1, 90)       # риска
        c -= edge_chamfer(x0, r_in, EF, +1, True)                            # торец на столе
        c -= vcut(x0, +1, W_NOTCH_B, EF, R_SLV_OUT - 1, R_NOTCH_B, 180)
        c -= rad_box(x0 - 1, x0 + 0.6, 1.2, R_SLV_OUT - 1, R_OUT + 1, 0)    # вентиляция
        c.label = "collar"
        return c

    # ------------------------------------------------------------ тубус A
    def tab_a(self, gx):
        r0 = R_CORE - 1.5
        x0, x1 = gx + TAB_A_X0, gx + TAB_A_X1
        return xr_prism([(x0, r0), (x0, R_TAB_A), (x1, R_TAB_A), (x1 + (R_TAB_A - r0), r0)], W_A)

    def core_tube(self):
        L = self.l_core
        t = extrude_x(bd.Circle(R_CORE) - u_profile() - cargo_profile(), 0, L)
        for gx in self.gx:
            t += self.tab_a(gx)
        # правый конец стоит на столе: фаски + вентиляция тайника в канал
        t -= edge_chamfer(L, R_CORE, EF, -1, False)
        t -= extrude_x(u_profile(EF), L, L - EF, taper=45) + extrude_x(u_profile(EF), L, L + 1)
        t -= bd.Pos(L, 0, -5.5) * bd.Box(1.6, 1.5, 4.0)
        if self.figure:                                   # штырьки под торец-подставку
            for a in PIN_ANGLES:
                t += rot(a) * cyl(PIN_R, -PIN_L, 0.5, z=PIN_RAD)
        t.label = "core_A_tube"
        return t

    # ------------------------------------------------------------ капсула B
    def head_b(self, gx):
        top, x0, x1 = R_HEAD_B1, gx + HEAD_B_X0, gx + HEAD_B_X1

        def ramp(r):
            return x0 - (top - r)
        rs = R_FIN_TOP - 0.5
        wings = [(ramp(R_HEAD_B0), R_HEAD_B0), (x0, top), (x1, top), (x1, R_HEAD_B0)]
        stem = [(ramp(rs), rs), (x0, top), (x1, top), (x1, rs)]
        return xr_prism(wings, W_B, 180) + xr_prism(stem, W_STEM, 180)

    def capsule_tube(self):
        x0, x1 = self.xc0, self.xc1
        c = cyl(R_CAPS, x0, x1, z=Z_CAPS)
        rb = -Z_CAPS + 1.5
        c += xr_prism([(self.x_fin0 - (R_FIN_TOP - rb), rb), (self.x_fin0, R_FIN_TOP),
                       (x1, R_FIN_TOP), (x1, rb)], W_STEM, 180)
        for gx in self.gx:
            c += self.head_b(gx)
        c -= cyl(R_CAPS_IN, x0 - 1, x1 + 1, z=Z_CAPS)                     # канал свитка
        # ключ под ребро бобышки торца (правый конец) и вентиляция у стола (левый конец)
        c -= bd.Pos((x1 - BOSS_H - 0.4 + x1 + 1) / 2, 0, Z_CAPS - 3.7) * bd.Box(BOSS_H + 1.4, 1.6, 1.4)
        c -= bd.Pos(x0 + 0.6, 0, Z_CAPS + 4.0) * bd.Box(1.2 + 0.01, 1.5, 2.0)
        c.label = "capsule_B_tube"
        return c

    # ------------------------------------------------------------ торцы (одинаковые снаружи)
    def cap_disc(self, x_in, outward):
        x_out = x_in + outward * self.cap_t
        x0, x1 = min(x_in, x_out), max(x_in, x_out)
        c = cyl(R_OUT, x0, x1)
        c -= edge_chamfer(x_out, R_OUT, 1.2, -outward, False)
        c -= edge_chamfer(x_in, R_OUT, 0.5, outward, False)
        for r in (19.0, 12.0):                                   # декоративные кольца
            g0, g1 = (x_out - 1, x_out + 0.8) if outward < 0 else (x_out - 0.8, x_out + 1)
            c -= tube(r - 0.5, r + 0.5, g0, g1)
        for k in range(36):                                      # насечка
            c -= rot(k * 10) * rad_box(x0 + 1.5, x1 - 1.5, 1.2, R_OUT - 0.5, R_OUT + 1)
        return c

    def cap_a_figure(self):
        """Торец-подставка фигурки: гладкий диск Ø50 со скруглением сверху (как у её подставки).
        Гнёзда под кроксы вычитаются потом из сетки (surgeon_sockets.py)."""
        t = CAP_T_FIG
        c = cyl(R_OUT, -t, 0)
        c = bd.fillet(c.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.X)[0], 2.5)
        c -= edge_chamfer(0.0, R_OUT, EF, -1, False)          # внутренняя сторона лежит на столе
        for a in PIN_ANGLES:
            c -= rot(a) * cyl(PIN_R + CLR_GLUE, -PIN_L - 0.5, 0.5, z=PIN_RAD)
        c.label = "cap_A"
        return c

    def cap_a(self):
        if self.figure:
            return self.cap_a_figure()
        c = self.cap_disc(0.0, -1) + extrude_x(cargo_profile(-CLR_GLUE), 0, BOSS_H)
        c.label = "cap_A"
        return c

    def cap_b(self):
        x1 = self.xc1
        boss = extrude_x(bd.Pos(0, Z_CAPS) * bd.Circle(R_CAPS_IN - CLR_GLUE), x1, x1 - BOSS_H)
        rib = bd.Pos(x1 - BOSS_H / 2, 0, Z_CAPS - 3.6) * bd.Box(BOSS_H, 1.6 - 2 * CLR_GLUE, 1.3)
        c = self.cap_disc(x1, +1) + boss + rib
        c.label = "cap_B"
        return c

    # ------------------------------------------------------------ сборки (для проверок)
    def core_assembled(self):
        return self.core_tube() + self.cap_a()

    def capsule_assembled(self):
        return self.capsule_tube() + self.cap_b()

    # ------------------------------------------------------------ ориентация для печати
    def print_parts(self, letters=True):
        """Детали, повёрнутые для печати: ось Z вверх, низ на Z=0 (стол), центр в XY=0."""
        down_left = bd.Rot(0, -90, 0)       # x -> Z (левый торец на столе)
        down_right = bd.Rot(0, 90, 0)       # x -> -Z (правый торец на столе)
        parts = {}
        parts["sleeve"] = down_left * self.sleeve()
        parts["collar"] = bd.Pos(0, 0, -self.x_collar) * (down_left * self.collar())
        parts["core_A_tube"] = bd.Pos(0, 0, self.l_core) * (down_right * self.core_tube())
        brim = cyl(9.0, self.xc0 - 0.8, self.xc0 + 0.01, z=Z_CAPS)  # отламываемая «юбка» под тонкую капсулу
        parts["capsule_B_tube"] = bd.Pos(0, 0, -(self.xc0 - 0.8)) * (down_left * (self.capsule_tube() + brim))
        if self.figure:      # гладкой внутренней стороной на стол, гнёзда под фигурку вверх
            parts["cap_A"] = down_right * self.cap_a()
        else:
            parts["cap_A"] = bd.Pos(0, 0, self.cap_t) * (down_left * self.cap_a())
        parts["cap_B"] = bd.Pos(0, 0, self.xc1 + self.cap_t) * (down_right * self.cap_b())
        for i in range(self.n):
            parts[f"ring_{i + 1}"] = down_left * self.ring(i, letters)
        out = {}
        for name, s in parts.items():
            bb = s.bounding_box()
            s = bd.Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s
            s.label = name
            out[name] = s
        return out


def export(out_dir: Path, word_a=WORD_A, word_b=WORD_B, cap_t=CAP_T):
    out_dir.mkdir(parents=True, exist_ok=True)
    cx = Cryptex(word_a, word_b, cap_t)
    total = 0.0
    for name, s in cx.print_parts().items():
        bb = s.bounding_box()
        print(f"{name:16s} {s.volume / 1000:6.1f} мл  {bb.size.X:5.1f}×{bb.size.Y:5.1f}×{bb.size.Z:5.1f} мм"
              f"  valid={s.is_valid}")
        total += s.volume
        bd.export_stl(s, str(out_dir / f"{name}.stl"), tolerance=0.01, angular_tolerance=0.1)
    print(f"итого смолы ~{total / 1000:.0f} мл")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        export(HERE.parent / "cryptex_test", WORD_A[-1], WORD_B[-1], cap_t=3.5)
    else:
        export(HERE)
