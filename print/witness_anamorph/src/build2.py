"""Witness-анаморфоза v2: игровая шкатулка (рельеф 4 граней со скринов) + двухслойная рваная картина.

Сцена, мм. y — вверх, z — от глаза вглубь, x — влево для зрителя (правая тройка; камера смотрит в +z).
Глаз E=(0, YE, -D). Передний слой картины z=0..T, задний z=T+G..2T+G. Передняя грань шкатулки z=ZF.
"""
import os, sys, json, time
import numpy as np, cv2, trimesh, manifold3d as M
from scipy import ndimage as ndi
from skimage.measure import find_contours

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out2'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.1f}s]', *a, flush=True)

# ---------------- параметры ----------------
ASP = 1.28                    # ширина/высота грани (калибровка по скрину 2)
WF = 46.0; HF = WF / ASP      # грань рельефа, мм
HB = WF / 2
Y0 = 8.0; Y1 = Y0 + HF        # пояс рельефа по высоте
D, T, G = 90.0, 1.2, 18.0     # глаз->картина, толщина слоя, зазор слоёв
ZF = 29.0                     # передняя грань шкатулки
REL = 0.5                     # высота белого рельефа
WALL = 2.0                    # стенка корпуса шкатулки
CL = 0.15                     # зазор посадок (смола)
FRES = 0.03                   # растр граней, мм
PRES = 0.03                   # растр плоскости картины, мм
FW, FH = 2048, 1600           # растр выпрямленных текстур

# ---------------- фигура на грани (из скрина 7 через гомографию) ----------------
Hf = np.load('H_7toface.npy')
ap = np.load('fig_mask.npz')['ap'].astype(np.float32)
ap = cv2.GaussianBlur(ap, (0, 0), 0.7)
Sface = cv2.warpPerspective(ap, Hf, (FW, FH), flags=cv2.INTER_LINEAR)   # 0..1, u вправо, v вниз
ys_, xs_ = np.where(Sface > 0.5)
u_c = (xs_.min() + xs_.max()) / 2 / FW; v_c = (ys_.min() + ys_.max()) / 2 / FH
XB = u_c * WF - HB            # центр шкатулки по x: ось глаза проходит через центр фигуры
YE = Y1 - v_c * HF            # высота глаза
ZB = ZF + HB                  # центр шкатулки по z
K = D / (D + ZF)              # масштаб грань -> плоскость картины
log(f'фигура u {xs_.min()/FW:.3f}..{xs_.max()/FW:.3f}  v {ys_.min()/FH:.3f}..{ys_.max()/FH:.3f}; XB={XB:.2f} YE={YE:.2f} K={K:.4f}')

# ---------------- плоскость картины: растр ----------------
PX0, PX1, PY0_, PY1_ = -34.0, 34.0, 0.0, 60.0
xs = np.arange(PX0, PX1, PRES) + PRES / 2
ys = np.arange(PY0_, PY1_, PRES) + PRES / 2
GX, GY = np.meshgrid(xs, ys)                      # строка = y (вверх), столбец = x
# точка картины -> точка грани -> пиксели текстуры
fx = GX / K; fy = YE + (GY - YE) / K
U = (XB + HB - fx) / WF * FW; V = (Y1 - fy) / HF * FH
Scont = cv2.remap(Sface, U.astype(np.float32), V.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
S = Scont > 0.5
uu = (XB + HB - fx) / WF
S &= uu < 1 - 0.25 / WF                           # справа за ребром — пустота, отступ 0.25 мм
S = ndi.binary_fill_holes(S)                      # островов внутри силуэта быть не может
S = ndi.binary_opening(S, structure=np.ones((3, 3)), iterations=2)   # щели < ~0.15 мм не печатаются
log('площадь силуэта на картине, мм2:', round(S.sum() * PRES**2, 1))

def noise(scale_mm, seed):
    r = np.random.default_rng(seed).standard_normal(GX.shape).astype(np.float32)
    n = ndi.gaussian_filter(r, scale_mm / PRES); return n / n.std()

dist = ndi.distance_transform_edt(~S) * PRES
R = 1.0 + 3.6 * np.clip(0.5 + 0.4 * noise(3.0, 1), 0, 1) + 0.35 * noise(0.35, 2)
Ug = dist < R
N = noise(1.5, 3)
GAP = 0.3
A1 = Ug & ~S & (N > GAP)
A2 = Ug & ~S & (N < -GAP)
def blob(cx, cy, r, seed):
    d = np.hypot(GX - cx, GY - cy) / r + 0.35 * noise(0.8, seed)
    return d < 1
A1 |= blob(22, 46, 3.2, 11) | blob(-21, 11, 2.3, 12) | blob(-24, 40, 1.4, 15)
A2 |= blob(-22, 45, 2.7, 13) | blob(21, 12, 2.5, 14) | blob(25, 30, 1.3, 16)
A1 &= ~S; A2 &= ~S & ~A1

FR_X, FR_Y0, FR_Y1 = 33.0, 2.0, 58.0       # наружный размер картины
FRW = 2.0                                  # стенка рамы
inner = (np.abs(GX) < FR_X - FRW - 1.0 - 0.6) & (GY > FR_Y0 + FRW + 1.0 + 0.6) & (GY < FR_Y1 - FRW - 1.0 - 0.6)
A1 &= inner; A2 &= inner
assert (S & ~inner).sum() == 0, 'фигура вылезла за поле картины'

disk = lambda r_mm: (lambda r: (np.hypot(*np.mgrid[-r:r+1, -r:r+1]) <= r))(int(round(r_mm / PRES)))
THIN = disk(0.35)   # перемычки уже ~0.7 мм убираем
def fix_layers(iters=8):
    global A1, A2
    A1_fixed = np.zeros_like(A1)
    for it in range(iters):
        ch = 0
        for k in (1, 2):
            Hk = S | (A1 if k == 1 else A2)
            sol = ~Hk
            # острова
            lab, n = ndi.label(sol)
            border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
            isl = (lab > 0) & ~np.isin(lab, border)
            # тонкие перемычки (только внутри поля картины)
            thin = sol & ~ndi.binary_opening(sol, structure=THIN) & inner
            bad = isl | thin
            if k == 2: bad &= ~A1_fixed
            if bad.any():
                ch += int(bad.sum())
                if k == 1: A1 |= bad; A2 &= ~bad; A1_fixed |= bad
                else: A2 |= bad; A1 &= ~bad
        if ch == 0: break
        log(f'  чистка слоёв, проход {it}: исправлено {ch} пикс')
fix_layers()
H1 = S | A1; H2 = S | A2
assert ((H1 & H2) == S).all()
log('дыры готовы: передний', round(H1.sum() * PRES**2, 1), 'мм2, задний', round(H2.sum() * PRES**2, 1), 'мм2')

# ---------------- геометрия: утилиты ----------------
def cs_from_field(field, x0, y0, res, level=0.5, eps=0.012):
    pad = np.pad(field.astype(np.float32), 2, constant_values=0)
    cs = find_contours(pad, level)
    polys = [np.c_[x0 + (c[:, 1] - 2 + 0.5) * res, y0 + (c[:, 0] - 2 + 0.5) * res][::-1] for c in cs if len(c) > 5]
    return M.CrossSection(polys, M.FillRule.EvenOdd).simplify(eps)

def smooth_mask(m, sig_px=1.0):
    return ndi.gaussian_filter(m.astype(np.float32), sig_px)

def cone_hole(cs, za, zb):
    """Дыра строго по лучам из глаза: стенки дыры направлены на глаз и из него не видны."""
    k0 = (za + D) / D
    c = cs.translate((0, -YE)).scale((k0, k0))
    kt = (zb + D) / (za + D)
    return M.Manifold.extrude(c, zb - za, 0, 0, (kt, kt)).translate((0, YE, za))

def box(x0, x1, y0, y1, z0, z1):
    return M.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))

def sqy(h, ya, yb, cx=0.0, cz=0.0):   # квадрат в плане (x,z) полуразмера h, по высоте ya..yb
    return box(cx - h, cx + h, ya, yb, cz - h, cz + h)

# ---------------- картина ----------------
csH1 = cs_from_field(smooth_mask(H1), PX0, PY0_, PRES)
csH2 = cs_from_field(smooth_mask(H2), PX0, PY0_, PRES)
ZB1 = T + G
front = box(-FR_X, FR_X, FR_Y0, FR_Y1, 0, T)
front += box(-FR_X, FR_X, FR_Y0, FR_Y1, 0, ZB1 + T) - box(-FR_X + FRW, FR_X - FRW, FR_Y0 + FRW, FR_Y1 - FRW, -1, ZB1 + T + 1)
front += box(-FR_X, FR_X, FR_Y0, FR_Y1, T, ZB1) - box(-FR_X + FRW + 1, FR_X - FRW - 1, FR_Y0 + FRW + 1, FR_Y1 - FRW - 1, 0, ZB1 + 1)
front += box(-FR_X, FR_X, FR_Y0, FR_Y1, -1.0, 0.1) - box(-FR_X + 3, FR_X - 3, FR_Y0 + 3, FR_Y1 - 3, -2, 1)          # багет
front += box(-FR_X + 0.8, FR_X - 0.8, FR_Y0 + 0.8, FR_Y1 - 0.8, -1.6, -0.9) - box(-FR_X + 2.2, FR_X - 2.2, FR_Y0 + 2.2, FR_Y1 - 2.2, -2, 1)
FEET = [(-24.0, -16.0), (16.0, 24.0)]
for a, b in FEET: front += box(a, b, -3.0, FR_Y0 + 0.5, 0, ZB1 + T)
front -= cone_hole(csH1, -2.0, T + 0.3)
back = box(-FR_X + FRW + CL, FR_X - FRW - CL, FR_Y0 + FRW + CL, FR_Y1 - FRW - CL, ZB1, ZB1 + T)
back -= cone_hole(csH2, ZB1 - 0.3, ZB1 + T + 0.3)
back = max(back.decompose(), key=lambda m_: m_.volume())
front = max(front.decompose(), key=lambda m_: m_.volume())
log('картина готова')

# ---------------- шкатулка ----------------
feat = np.load('faces_feat.npz')
NU, NV = int(round(WF / FRES)), int(round(HF / FRES))
def face_mask(key, fix_red=False):
    f = feat[key].astype(np.float32)
    if fix_red:   # красный кабель лазера на грани с дверью: заполняем ближайшим рельефом
        r = cv2.imread(f'rect_{key}.png'); lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB).astype(float)
        red = ndi.binary_dilation((lab[..., 1] - 128 > 30) & (lab[..., 0] > 40), iterations=6)
        idx = ndi.distance_transform_edt(red, return_distances=False, return_indices=True)
        f = f[idx[0], idx[1]]
    f = cv2.resize(f, (NU, NV), interpolation=cv2.INTER_AREA)
    Wm = f < 0                                       # белое (выпуклое)
    Wm = ndi.binary_opening(Wm, iterations=3); Wm = ndi.binary_closing(Wm, iterations=3)
    amin = int((0.35 / FRES) ** 2)
    lab, n = ndi.label(Wm); sz = ndi.sum(Wm, lab, range(1, n + 1)); Wm &= np.isin(lab, 1 + np.where(sz >= amin)[0])
    lab, n = ndi.label(~Wm); sz = ndi.sum(~Wm, lab, range(1, n + 1)); Wm |= np.isin(lab, 1 + np.where(sz < amin)[0])
    return Wm   # строки: v сверху вниз, столбцы: u слева направо (вид снаружи)

def face_frame(nrm):
    n = np.array(nrm, float); u_dir = np.cross(-n, [0, 1, 0])
    c = np.array([XB, 0, ZB]) + n * HB
    origin = c - u_dir * HB
    Mx = np.c_[u_dir, [0, 1, 0], n, origin]
    return Mx, u_dir, n, origin

FACES = {'front': ('f2', (0, 0, -1)), 'left_x+': ('f3R', (1, 0, 0)), 'back': ('f3L', (0, 0, 1)), 'right_x-': ('f1', (-1, 0, 0))}
masks = {}
relief = []
for name, (key, nrm) in FACES.items():
    Wm = face_mask(key, fix_red=(key == 'f3L'))
    masks[name] = Wm
    cs = cs_from_field(smooth_mask(Wm[::-1], 1.2), 0.0, Y0, FRES, eps=0.012)
    rel = M.Manifold.extrude(cs, REL + 0.3).translate((0, 0, -0.3))
    Mx, *_ = face_frame(nrm)
    relief.append(rel.transform(Mx))
    log('рельеф грани', name, key, 'белого', round(Wm.mean() * 100), '%')

core = sqy(HB, Y0 - 0.01, Y1 + 0.01, XB, ZB)
body = core
for r in relief: body += r
# корпус (профиль по скринам: 3 ступени цоколя, 3 ступени карниза, верх — усечённая пирамида)
shell = (sqy(28.0, 0, 3.5, XB, ZB) + sqy(26.0, 3.5, 6.5, XB, ZB) + sqy(24.2, 6.5, Y0, XB, ZB)
         + sqy(24.2, Y1, Y1 + 1.5, XB, ZB) + sqy(26.0, Y1 + 1.5, Y1 + 3.7, XB, ZB) + sqy(27.5, Y1 + 3.7, Y1 + 5.7, XB, ZB))
YT0 = Y1 + 5.7; YT1 = 60.0
top = M.Manifold.extrude(M.CrossSection.square((40.0, 40.0), True), YT1 - 1.6 - YT0, 0, 0, (33.0 / 40, 33.0 / 40))
top = top.rotate((-90, 0, 0)).translate((XB, YT0, ZB))
top += sqy(15.5, YT1 - 1.6, YT1, XB, ZB)
box_s = body + shell + top
# дверь на задней грани (по скрину 3), решётка и пульт
Mx, u_dir, n, origin = face_frame((0, 0, 1))
def on_face(u0, u1, v0, v1, w0, w1):
    loc = box(u0 * WF, u1 * WF, Y1 - v1 * HF, Y1 - v0 * HF, w0, w1)
    return loc.transform(Mx)
DU0, DU1, DV0, DV1 = 0.333, 0.597, 0.231, 1.0
box_s -= on_face(DU0, DU1, DV0, DV1 + 0.05, -0.8, 1.0)
bars = []
pitch = 1.45
uA, uB = DU0 * WF, DU1 * WF; yA, yB = Y1 - DV1 * HF, Y1 - DV0 * HF
for x in np.arange(uA + 0.6, uB - 0.3, pitch): bars.append(box(x, x + 0.45, yA, yB, -0.8, -0.3))
for y in np.arange(yA + 0.8, yB - 0.3, pitch): bars.append(box(uA, uB, y, y + 0.45, -0.8, -0.3))
lattice = M.Manifold.batch_boolean(bars, M.OpType.Add).transform(Mx)
box_s += lattice
fr = 0.018
box_s += on_face(DU0 - fr, DU1 + fr, DV0 - fr * ASP, DV1, -0.8, 0.45) - on_face(DU0, DU1, DV0, DV1 + 0.1, -1, 1)
PU0, PU1, PV0, PV1 = 0.565, 0.665, 0.70, 0.985
box_s += on_face(PU0, PU1, PV0, PV1, -0.8, 1.3) - on_face(PU0 + 0.008, PU1 - 0.008, PV0 + 0.01, PV1 - 0.01, 1.0, 2.0)
# полость: открытое дно, стенки >= 2 мм
cav = sqy(25.8 - 2.2, -1, 2.0, XB, ZB) + sqy(HB - WALL, -1, Y1 + 3.0, XB, ZB)
cav += M.Manifold.extrude(M.CrossSection.square((36.0, 36.0), True), 6.0, 0, 0, (29.0 / 36, 29.0 / 36)).rotate((-90, 0, 0)).translate((XB, Y1 + 3.0, ZB))
box_s -= cav
log('шкатулка готова')

# ---------------- подставка, штанга, глазок ----------------
BX0, BX1 = -36.0, XB + 31.0
BZ0, BZ1 = -4.0, ZB + 31.0
BT = 6.0
base = box(BX0, BX1, -BT, 0, BZ0, BZ1)
for a, b in FEET: base -= box(a - CL, b + CL, -3.2, 1, -CL, ZB1 + T + CL)
base -= sqy(28.0 + CL, -1.0, 1, XB, ZB)
TW, TY0, TY1, TL = 8.0, -4.5, -1.5, 20.0     # язычок штанги: полуширина, низ, верх, длина
base -= box(-TW - 0.1, TW + 0.1, TY0 - 0.1, TY1 + 0.1, BZ0 - 1, BZ0 + TL + 0.2)
# штанга с глазком
arm = box(-TW, TW, -BT, 0, -D - 6.0, BZ0)
arm += box(-TW, TW, TY0, TY1, BZ0 - 0.5, BZ0 + TL)
PH = 14.0
post = box(-PH, PH, -BT, YE + PH - 2, -D, -D + 2.4)
post -= box(-5, 5, YE - 5, YE + 5, -D - 1, -D + 3)                       # окно
post -= box(-12.0, 12.0, YE - 12.0, YE + 12.0, -D - 1, -D + 0.8)         # гнездо карточки 24x24x0.8
arm += post
arm += box(-1.2, 1.2, 0, 12, -D + 2.4, -D + 12)                           # ребро
arm += box(-PH, PH, -BT, 0, -D - 6.0, -D + 2.4)
def card(d):
    c = box(-10.5, 10.5, YE - 10.5, YE + 10.5, -D, -D + 0.8)
    c -= M.Manifold.cylinder(3, d / 2, d / 2, 64).translate((0, YE, -D - 1))
    # риски-перекрестье для ориентира
    for dx in (-8.5, 8.5): c -= box(dx - 0.2, dx + 0.2, YE - 1.5, YE + 1.5, -D - 1, -D + 0.25)
    for dy in (-8.5, 8.5): c -= box(-1.5, 1.5, YE + dy - 0.2, YE + dy + 0.2, -D - 1, -D + 0.25)
    return c
card_eye, card_phone = card(1.5), card(3.0)
log('подставка готова')

parts = {'painting_front': front, 'painting_back': back, 'box': box_s, 'base': base, 'eyepiece_arm': arm,
         'eyepiece_card_eye_1.5mm': card_eye, 'eyepiece_card_phone_3mm': card_phone}

def tm(m_):
    mg = m_.to_mesh(); return trimesh.Trimesh(np.asarray(mg.vert_properties)[:, :3], np.asarray(mg.tri_verts), process=False)
info = {}
for k_, m_ in parts.items():
    t_ = tm(m_); t_.export(f'{OUT}/scene_{k_}.stl')
    info[k_] = dict(watertight=bool(t_.is_watertight), bodies=len(m_.decompose()), vol_cm3=round(float(t_.volume) / 1000, 2),
                    size=[round(float(e), 1) for e in t_.extents], tris=len(t_.faces))
# для рендера: «бежевое» и «белое» раздельно
tm(box_s - M.Manifold.batch_boolean(relief, M.OpType.Add) + core).export(f'{OUT}/render_box_stone.stl')
tm(M.Manifold.batch_boolean(relief, M.OpType.Add) - core).export(f'{OUT}/render_box_white.stl')
geo = dict(D=D, T=T, G=G, ZF=ZF, XB=XB, YE=YE, ZB=ZB, WF=WF, HF=HF, Y0=Y0, Y1=Y1, K=K, REL=REL,
           PX0=PX0, PY0=PY0_, PRES=PRES, FR=[FR_X, FR_Y0, FR_Y1], info=info)
json.dump(geo, open(f'{OUT}/geo.json', 'w'), indent=1)
np.savez_compressed(f'{OUT}/plane.npz', S=S, A1=A1, A2=A2)
np.savez_compressed(f'{OUT}/front_face.npz', W=masks['front'])
log(json.dumps(info, ensure_ascii=False, indent=1))
