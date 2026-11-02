"""Криптекс на 5 свитков: 5 слов — 5 секций. Под смолу, все детали печатаются без поддержек.

Внутри жёсткая рама (ось + 5 перегородок + гильза с прорезями + левый фланец),
между перегородками 5 выдвижных секций-клиньев со свитками. У каждой секции своя линия
зубьев и свой профиль зуба: от узкого и высокого (секция 1) до широкого и низкого (секция 5).
В каждом кольце 5 пазов таких же профилей; зуб проходит только в свой паз
(шире — не влезает в узкий, выше — не влезает в мелкий).
Секции 1–3 выезжают влево, 4–5 — вправо.

Оси и углы как в cryptex.py: ось X, угол a — от +Z к -Y, линия чтения a=90 (спереди).
    python cryptex5.py   (cadgen / build123d)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cryptex"))
from cadgen import build123d as bd  # noqa: E402
from cryptex import (rot, cyl, tube, rad_box, xr_prism, revolve_xr, edge_chamfer,  # noqa: E402
                     vcut, gable, extrude_x, ang_dist)

FONT = str(HERE.parent / "cryptex" / "fonts" / "DejaVuSerif-Bold.ttf")
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N = 26
STEP = 360.0 / N

# секции: слово, угол линии зубьев, сторона выхода (-1 влево, +1 вправо)
WORDS = ["SOFIA", "ROMAN", "CROSS", "APPLE", "TOWER"]
THETA = [0.0, 144.0, 288.0, 72.0, 216.0]
EXIT = [-1, -1, -1, +1, +1]
W_HEAD = [2.4, 3.6, 4.8, 6.0, 7.2]      # ширина головки зуба
H_TAB = [4.4, 3.6, 2.8, 2.0, 1.2]       # высота зуба над гильзой
W_STEM = 2.4

# зазоры (смола) — как в cryptex.py
CLR, CLR_SLIDE, CLR_T, CLR_GLUE, EF, LEAD = 0.20, 0.25, 0.25, 0.15, 0.5, 0.6

R_HUB = 4.5
WEB = 1.6
R_SLV_IN, R_SLV_OUT = 16.0, 17.8
R_BORE = R_SLV_OUT + CLR
R_HEAD0 = 18.1
R_OUT = 27.5                              # Ø55
TOP = [R_SLV_OUT + h for h in H_TAB]
NOTCH_R = [t + 0.35 for t in TOP]
R_GROOVE = TOP[0] + 0.4
R_LEDGE = max(TOP[k] for k in range(5) if EXIT[k] > 0) + 0.4   # плоский уступ под правые зубья
WALL = 1.2                                 # стенка секции

RW, GW, GAP = 15.0, 9.2, 0.2
FL, COLLAR_W, PLATE_T, CAP_T = 5.0, 5.0, 2.0, 4.0
END_WALL = 1.5                             # заглушка левого конца правых карманов


def pt(r, a):
    return (-r * math.sin(math.radians(a)), r * math.cos(math.radians(a)))


def wedge(a, half):
    return bd.Polygon((0, 0), pt(80, a - half), pt(80, a), pt(80, a + half), align=None)


def annulus(r0, r1):
    return bd.Circle(r1) - bd.Circle(r0)


def pocket(k, grow=0.0):
    """Карман рамы под секцию k (grow<0 — сама секция)."""
    w = bd.offset(wedge(THETA[k], 36.0), -(WEB / 2) + grow)
    return annulus(R_HUB - grow, R_SLV_IN + grow) & w


def letter_cut(ch, x_c):
    t = bd.Text(ch, 7.0, font_path=FONT, align=(bd.Align.CENTER, bd.Align.CENTER))
    h = t.bounding_box().size.Y
    if h > 5.3:
        t = bd.scale(t, 5.3 / h)
    solid = bd.extrude(bd.Plane.XZ * t, amount=3.0)
    return bd.Pos(x_c, -(R_OUT - 0.6), 0) * solid


class Cryptex5:
    def __init__(self, words=WORDS):
        self.words = words
        self.n = len(words[0])
        self.x_ring0 = FL + GAP
        self.pitch = RW + GAP
        self.gx = [self.x_ring0 + i * self.pitch + RW / 2 for i in range(self.n)]
        self.x_collar = self.x_ring0 + self.n * self.pitch
        self.x_end = self.x_collar + COLLAR_W
        self.x_plate = self.x_end + PLATE_T
        self.check_words()

    # -------------------------------------------------------------- слова
    def notch_angle(self, ring, k):
        return THETA[k] - 90.0 + ALPHABET.index(self.words[k][ring]) * STEP

    def check_words(self):
        for r in range(self.n):
            for i in range(5):
                for j in range(i + 1, 5):
                    need = math.degrees(((W_HEAD[i] + W_HEAD[j]) / 2 + 0.5 + 2 * LEAD + 1.0) / R_BORE)
                    d = ang_dist(self.notch_angle(r, i), self.notch_angle(r, j))
                    assert d > need, (f"кольцо {r + 1}: пазы слов {self.words[i]} и {self.words[j]} "
                                      f"наложатся — поменяй букву")

    # -------------------------------------------------------------- зуб секции
    def tab(self, k, gx):
        top, a = TOP[k], THETA[k]
        if EXIT[k] < 0:      # выход влево: упор левой гранью, скос справа
            x0, x1 = gx - 3.5, gx - 1.0

            def prof(rb):
                return [(x0, rb), (x0, top), (x1, top), (x1 + (top - rb), rb)]
        else:                # выход вправо: упор правой гранью, скос слева
            x0, x1 = gx - 1.6, gx + 2.4

            def prof(rb):
                return [(x0 - (top - rb), rb), (x0, top), (x1, top), (x1, rb)]
        t = xr_prism(prof(R_SLV_IN - 1.4), W_STEM, a)
        if W_HEAD[k] > W_STEM:
            t += xr_prism(prof(R_HEAD0), W_HEAD[k], a)
        return t

    def tab_extent(self, k, r):
        """Крайняя координата зуба (от центра канавки) на радиусе r — для длины прорезей."""
        if EXIT[k] < 0:
            return -1.0 + (TOP[k] - r)
        return -1.6 - (TOP[k] - r)

    # -------------------------------------------------------------- рама
    def frame(self):
        sec = bd.Circle(R_SLV_OUT)
        for k in range(5):
            sec -= pocket(k)
        f = extrude_x(sec, 0, self.x_end)
        f += tube(R_SLV_IN, R_OUT, 0, FL)
        f -= edge_chamfer(0, R_OUT, 0.7, +1, False) + edge_chamfer(FL, R_OUT, 0.7, -1, False)
        for k in range(5):
            a, wn = THETA[k], W_HEAD[k] + 2 * CLR_T
            if EXIT[k] < 0:
                f -= extrude_x(pocket(k), -1, FL + 1)                    # карман открыт слева (и во фланце)
                xe = self.gx[-1] + self.tab_extent(k, R_SLV_IN) + 0.4
                f -= rad_box(-1, xe, W_STEM + 2 * CLR_T, R_SLV_IN - 1.5, R_SLV_OUT + 1, a)
                f -= gable(xe, W_STEM + 2 * CLR_T, R_SLV_IN - 1.5, R_SLV_OUT + 1, a)
                f -= rad_box(-1, FL + 1, wn, R_SLV_IN - 1.5, NOTCH_R[k], a)  # ворота во фланце
                f -= vcut(0, +1, wn, EF, R_SLV_IN - 1.5, NOTCH_R[k], a)
                f -= extrude_x(pocket(k, EF), 0, EF, taper=45)          # фаска устья у стола
            else:
                f += extrude_x(pocket(k), 0, END_WALL)                  # левый конец кармана закрыт
                xs = self.gx[0] + self.tab_extent(k, R_SLV_IN) - 0.4
                f -= rad_box(xs, self.x_end + 1, W_STEM + 2 * CLR_T, R_SLV_IN - 1.5, R_SLV_OUT + 1, a)
        f -= rad_box(-1, FL + 1, 1.0, R_OUT - 0.6, R_OUT + 1, 90)         # риска линии чтения
        f.label = "frame"
        return f

    def end_plate(self):
        """Правая плита: упор колец (клеится на гильзу) + крышка карманов 1–3 + окна секций 4–5."""
        p = tube(R_SLV_OUT + 0.1, R_OUT, self.x_collar, self.x_end)
        p += cyl(R_OUT, self.x_end, self.x_plate)
        p -= edge_chamfer(self.x_collar, R_OUT, 0.7, +1, False) + edge_chamfer(self.x_plate, R_OUT, 0.7, -1, False)
        for k in range(5):
            if EXIT[k] > 0:
                p -= extrude_x(pocket(k), self.x_end - 1, self.x_plate + 1)
                p -= rad_box(self.x_collar - 1, self.x_plate + 1, W_HEAD[k] + 2 * CLR_T, R_SLV_IN - 1.5, NOTCH_R[k], THETA[k])
                p -= extrude_x(pocket(k, EF), self.x_plate, self.x_plate - EF, taper=45)
                p -= vcut(self.x_plate, -1, W_HEAD[k] + 2 * CLR_T, EF, R_SLV_IN - 1.5, NOTCH_R[k], THETA[k])
        p -= rad_box(self.x_collar - 1, self.x_plate + 1, 1.0, R_OUT - 0.6, R_OUT + 1, 90)
        p -= edge_chamfer(self.x_collar, R_SLV_OUT + 0.1, EF, +1, True)
        # вентиляция чаши при печати (плита на столе): канавка от окна 4 к краю, выглядит как метка
        p -= rad_box(self.x_plate - 0.6, self.x_plate + 1, 1.2, 10.0, R_OUT + 1, 85.0)
        p.label = "end_plate"
        return p

    # -------------------------------------------------------------- секции
    def drawer(self, k):
        sec = pocket(k, -CLR_SLIDE)
        cav = bd.offset(bd.offset(bd.offset(sec, -WALL), -0.6), 0.6)
        if EXIT[k] < 0:
            x_back, x_face = self.x_end - 0.3, -CAP_T
            body = extrude_x(sec, x_face, x_back) - extrude_x(cav, x_face + 3.0, x_back + 1)
            vent_x = x_back - 0.6
        else:
            x_back, x_face = END_WALL + 0.3, self.x_plate + CAP_T
            body = extrude_x(sec, x_back, x_face) - extrude_x(cav, x_back - 1, x_face - 3.0)
            vent_x = x_back + 0.6
        for gx in self.gx:
            body += self.tab(k, gx)
        # вентиляция полости у стола (со стороны оси, скрыта)
        body -= rot(THETA[k]) * (bd.Pos(vent_x, 0, R_HUB + 1.0) * bd.Box(1.2, 1.5, 4.0))
        # ручка на торце
        ridge_x = x_face - 1.2 if EXIT[k] < 0 else x_face + 1.2
        ridge = rot(THETA[k]) * (bd.Pos(ridge_x, 0, 11.0) * bd.Box(2.4, 2.2, 7.0))
        ridge = ridge - rot(THETA[k]) * (bd.Pos(ridge_x + EXIT[k] * 1.2, 0, 14.5) * bd.Rot(0, 45, 0) * bd.Box(2.4, 3, 2.4))
        body += ridge
        body.label = f"drawer_{k + 1}"
        return body

    # -------------------------------------------------------------- кольцо
    def ring(self, i, letters=True):
        xl, xr = (RW - GW) / 2, (RW + GW) / 2
        body = tube(R_BORE, R_OUT, 0, RW)
        body -= revolve_xr([(xl, R_BORE - 1), (xl, R_GROOVE), (xr - (R_GROOVE - R_LEDGE), R_GROOVE),
                            (xr, R_LEDGE), (xr, R_BORE - 1)])
        angs = []
        for k in range(5):
            a, wn = self.notch_angle(i, k), W_HEAD[k] + 2 * CLR_T
            angs.append(a)
            body -= rad_box(-1, RW + 1, wn, R_BORE - 1, NOTCH_R[k], a)
            body -= vcut(0, +1, wn, EF, R_BORE - 1, NOTCH_R[k], a)
            if EXIT[k] < 0:
                body -= vcut(xl, -1, wn, LEAD, R_BORE - 1, NOTCH_R[k], a)
            else:
                body -= vcut(xr, +1, wn, LEAD, R_BORE - 1, NOTCH_R[k], a)
        body -= edge_chamfer(0, R_BORE, EF, +1, True)
        # вентиляция у стола — в разделителях подальше от пазов
        cands = sorted(((min(ang_dist((m + 0.5) * STEP, a) for a in angs), (m + 0.5) * STEP) for m in range(N)), reverse=True)
        for _, a in cands[:3]:
            body -= rad_box(-1, 0.6, 1.2, R_BORE - 1, R_OUT + 1, a)
        body -= edge_chamfer(0, R_OUT, 0.7, +1, False) + edge_chamfer(RW, R_OUT, 0.7, -1, False)
        num = bd.Text(str(i + 1), 3.0, font_path=FONT, align=(bd.Align.CENTER, bd.Align.CENTER))
        face_r = bd.Plane(origin=(RW, 0, 0), x_dir=(0, -1, 0), z_dir=(-1, 0, 0))
        body -= bd.extrude(face_r * (bd.Pos(0, R_GROOVE + 2.4) * num), amount=0.4)
        if letters:
            for m, ch in enumerate(ALPHABET):
                body -= rot(m * STEP - 90.0) * letter_cut(ch, RW / 2)
            for m in range(N):
                body -= rot((m + 0.5) * STEP) * rad_box(1.2, RW - 1.2, 0.4, R_OUT - 0.4, R_OUT + 1)
        body.label = f"ring_{i + 1}"
        return body

    def ring_placed(self, i, code, ring_shape):
        m = ALPHABET.index(code[i])
        return bd.Pos(self.x_ring0 + i * self.pitch, 0, 0) * rot(90.0 - m * STEP) * ring_shape

    # -------------------------------------------------------------- печать
    def print_parts(self):
        dl, dr = bd.Rot(0, -90, 0), bd.Rot(0, 90, 0)
        parts = {"frame": dl * self.frame(), "end_plate": dr * self.end_plate()}
        for k in range(5):
            d = self.drawer(k)
            # секция печатается торцом-ручкой вверх, на отрезаемой «юбке»
            if EXIT[k] < 0:
                x_back = self.x_end - 0.3
                brim = extrude_x(bd.offset(pocket(k, -CLR_SLIDE), 3.0), x_back - 0.01, x_back + 0.8)
                parts[f"drawer_{k + 1}"] = dr * (d + brim)
            else:
                x_back = END_WALL + 0.3
                brim = extrude_x(bd.offset(pocket(k, -CLR_SLIDE), 3.0), x_back - 0.8, x_back + 0.01)
                parts[f"drawer_{k + 1}"] = dl * (d + brim)
        for i in range(self.n):
            parts[f"ring_{i + 1}"] = dl * self.ring(i)
        out = {}
        for name, s in parts.items():
            bb = s.bounding_box()
            s = bd.Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s
            s.label = name
            out[name] = s
        return out


if __name__ == "__main__":
    cx = Cryptex5()
    total = 0.0
    for name, s in cx.print_parts().items():
        bb = s.bounding_box()
        print(f"{name:10s} {s.volume / 1000:6.1f} мл {bb.size.X:5.1f}×{bb.size.Y:5.1f}×{bb.size.Z:5.1f} valid={s.is_valid}", flush=True)
        total += s.volume
        bd.export_stl(s, str(HERE / f"{name}.stl"), tolerance=0.01, angular_tolerance=0.1)
    print(f"итого ~{total / 1000:.0f} мл")
