"""Разборка «с решением», шаг 2: метки граней по цвету текстуры + складкам/швам.
ПРИМЕР для персонажа redhead_black_tracksuit — семена и константы (реглан, шея, волосы) подбираются под каждую фигурку.
usage: MODE=abs python3 seg_redhead.py K=40 model.glb [сглаж_вогнутости=4] [сглаж_границ=150]  -> L.npy, conc.npy в текущей папке
"""
import os, numpy as np, sys, trimesh, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_front, splat, classify
import os
MODE = os.environ.get('MODE', 'abs')
RULES = os.environ.get('RULES', '0') == '1'
K = float(sys.argv[1]) if len(sys.argv) > 1 else 40
m, fc = load_front(sys.argv[2] if len(sys.argv) > 2 else "../dl/detailed.glb")
cls = classify(m, fc)
C = m.triangles_center; z = C[:, 2]; nF = len(C)
# платформа: всё ниже верха платформы
up = (m.face_normals[:, 2] > 0.95) & (z < 15)
hz = np.histogram(z[up], bins=np.arange(0, 15, 0.1))
ztop = hz[1][np.argmax(hz[0])] + 0.05
print("верх платформы z=%.2f" % ztop)
cls[z < ztop + 0.15] = 4
# вогнутость (сглаженная)
adj = m.face_adjacency; ang = m.face_adjacency_angles.copy()
ang[m.face_adjacency_convex] *= -1          # + = вогнуто
A = sp.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), (nF, nF)).tocsr()
deg = np.asarray(A.sum(1)).ravel()
conc = np.zeros(nF); np.add.at(conc, adj[:, 0], ang); np.add.at(conc, adj[:, 1], ang)
conc /= np.maximum(deg, 1)
Dn = sp.diags(1 / np.maximum(deg, 1)) @ A
for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 15):
    conc = 0.5 * conc + 0.5 * (Dn @ conc)
np.save("conc.npy", conc)
d = np.linalg.norm(C[adj[:, 0]] - C[adj[:, 1]], axis=1)
cc = np.abs(conc[adj[:, 0]] + conc[adj[:, 1]]) / 2 if MODE == 'abs' else np.maximum(0, (conc[adj[:, 0]] + conc[adj[:, 1]]) / 2)
w = d * (1 + K * cc / max(np.percentile(cc[cc > 0], 90), 1e-6))
w[cls[adj[:, 0]] != cls[adj[:, 1]]] += 60.0   # смена цвета = дорого
G = sp.coo_matrix((np.r_[w, w], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), (nF, nF)).tocsr()

NAMES = ["platform", "shoe_left", "shoe_right", "pant_left", "pant_right", "body",
         "sleeve_left", "sleeve_right", "hands", "neck", "head", "hair"]
fr, idx = splat(m, fc, "front", 0.2)
bk, idxb = splat(m, fc, "back", 0.2)
def px(img_idx, c, r, want):
    # ближайший пиксель нужного цвета
    best = None
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            f = img_idx[r + dr, c + dc]
            if f >= 0 and cls[f] == want:
                dd = dr * dr + dc * dc
                if best is None or dd < best[0]: best = (dd, f)
    assert best, (c, r, want)
    return best[1]
F, B = idx, idxb
Wb = bk.shape[1]
seeds = {
 "platform": [px(F, 189, 487, 4), px(B, 180, 480, 4)],
 "shoe_left": [px(F, 138, 436, 2)], "shoe_right": [px(F, 262, 436, 2)],
 "pant_left": [px(F, 138, 371, 2)], "pant_right": [px(F, 262, 371, 2)],
 "body": [px(F, 247, 232, 2), px(B, Wb // 2, 280, 2)],
 "sleeve_left": [px(F, 124, 268, 2)], "sleeve_right": [px(F, 284, 254, 2)],
 "hands": [px(F, 196, 319, 3)],
 "head": [px(F, 211, 174, 3), px(F, 211, 140, 3)],
 "hair": [px(F, 117, 79, 1), px(B, Wb // 2, 80, 1)],
}
# шея: кожа рядом с воротником (касается чёрного, выше 50 мм)
black_nb = np.zeros(nF, bool)
e = (cls[adj[:, 0]] == 3) & (cls[adj[:, 1]] == 2); black_nb[adj[e, 0]] = True
e = (cls[adj[:, 1]] == 3) & (cls[adj[:, 0]] == 2); black_nb[adj[e, 1]] = True
neck = np.nonzero(black_nb & (z > 50) & (z < 60) & (np.abs(C[:, 0]) < 10))[0]
seeds["neck"] = list(neck)
print("шея: %d граней-семян, z %.1f..%.1f" % (len(neck), z[neck].min(), z[neck].max()))
XC, YC = 2.45, 1.0
P1z, P2z, P1x, P2x = 57.0, 38.0, 5.0, 15.0     # реглан: от воротника к подмышке
def rag(side):  # >0 = снаружи (рукав)
    dx = side * (C[:, 0] - XC)
    d = np.array([P2x - P1x, P2z - P1z]); n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return (dx - P1x) * n[0] + (C[:, 2] - P1z) * n[1]
rl, rr = rag(-1), rag(1)
blk = (cls == 2) & (C[:, 2] < 58)
dxc = np.abs(C[:, 0] - XC)
seeds["body"] += list(np.nonzero(blk & (dxc < 5) & (C[:, 2] > 40) & (C[:, 2] < 54) & (C[:, 1] > 4))[0])
seeds["body"] += list(np.nonzero(blk & (dxc < 4) & (C[:, 2] > 49) & (C[:, 2] < 55) & (C[:, 1] < -2))[0])
seeds["body"] += list(np.nonzero(blk & (dxc < 6) & (C[:, 1] > 12) & (C[:, 2] > 27) & (C[:, 2] < 37))[0])
seat = blk & (C[:, 1] > 8) & (C[:, 2] > 11) & (C[:, 2] < 16)
seeds["pant_left"] += list(np.nonzero(seat & (C[:, 0] < XC - 2))[0]); seeds["pant_right"] += list(np.nonzero(seat & (C[:, 0] > XC + 2))[0])
seeds["sleeve_left"] += list(np.nonzero(blk & (rl > 2.5) & (C[:, 2] > 44) & (C[:, 1] > -8) & (C[:, 0] < XC))[0])
seeds["sleeve_right"] += list(np.nonzero(blk & (rr > 2.5) & (C[:, 2] > 44) & (C[:, 1] > -8) & (C[:, 0] > XC))[0])
src, lab = [], []
for i, n in enumerate(NAMES):
    for f in seeds[n]:
        src.append(f); lab.append(i + 1)
src = np.array(src); lab = np.array(lab)
dist, pred, sources = dijkstra(G, directed=False, indices=src, min_only=True, return_predecessors=True)
L = np.zeros(nF, np.uint8)
ok = sources >= 0
lut = np.zeros(nF, np.uint8); lut[src] = lab
L[ok] = lut[sources[ok]]
print("не достигнуто:", (~ok).sum())
from scipy.spatial import cKDTree
if (~ok).any():
    t = cKDTree(C[ok]); _, j = t.query(C[~ok]); L[~ok] = L[ok][j]
ID = {n: i + 1 for i, n in enumerate(NAMES)}
# сзади шов реглана — по плоскости
bs = RULES & np.isin(L, [ID["body"], ID["sleeve_left"], ID["sleeve_right"]]) & (C[:, 1] > 2) & (C[:, 2] > 36)
L[bs] = ID["body"]
L[bs & (rl > 0) & (C[:, 0] < XC)] = ID["sleeve_left"]
L[bs & (rr > 0) & (C[:, 0] > XC)] = ID["sleeve_right"]
# сглаживание границ одежды (без пересечения цветов)
CL = [ID[n] for n in ["shoe_left", "shoe_right", "pant_left", "pant_right", "body", "sleeve_left", "sleeve_right"]]
inC = np.isin(L, CL)
e = inC[adj[:, 0]] & inC[adj[:, 1]]
Ac = sp.coo_matrix((np.ones(2 * e.sum()), (np.r_[adj[e, 0], adj[e, 1]], np.r_[adj[e, 1], adj[e, 0]])), (nF, nF)).tocsr()
dc = np.asarray(Ac.sum(1)).ravel(); Dc = sp.diags(1 / np.maximum(dc, 1)) @ Ac
oh = np.stack([(L == c).astype(np.float32) for c in CL], 1)
fixed = np.zeros(nF, bool); fixed[src] = True
for it in range(int(sys.argv[4]) if len(sys.argv) > 4 else 150):
    oh = 0.5 * oh + 0.5 * (Dc @ oh)
L[inC] = np.array(CL, np.uint8)[oh[inC].argmax(1)]
# шея — только цилиндр под подбородком, уши -> голова
nk = (L == ID["neck"]) & (np.hypot(C[:, 0] - XC, C[:, 1] - YC) > 8.5)
L[nk] = ID["head"]
# голова/волосы строго по цвету
from scipy.sparse.csgraph import connected_components
def comps(mask):
    e = mask[adj[:, 0]] & mask[adj[:, 1]]
    g = sp.coo_matrix((np.ones(e.sum()), (adj[e, 0], adj[e, 1])), (nF, nF))
    n, lb = connected_components(g, directed=False)
    lb = np.where(mask, lb, -1); return lb
r_, g_, b_ = fc.T
zone = np.isin(L, [ID["head"], ID["hair"], ID["neck"]])
clo = np.isin(L, [ID["body"], ID["sleeve_left"], ID["sleeve_right"]]) & (C[:, 2] > 50)
r_, g_, b_ = fc.T
rd = (r_ > g_ * 1.3) & (r_ > b_ * 1.3) & (r_ > 0.12)
lbx = comps((zone | clo) & rd)
hl = lbx[(L == ID["hair"]) & (lbx >= 0)]
if len(hl):
    zone = zone | (clo & rd & (lbx == np.bincount(hl).argmax()))
redish = zone & (r_ > g_ * 1.45) & (r_ > b_ * 1.45) & (r_ > 0.18)
lb = comps(redish)
sz = np.bincount(lb[lb >= 0]); hair = lb == np.argmax(sz)
# дыры в волосах (тени/блики, окружённые волосами) -> волосы
rest = zone & ~hair & (L != ID["neck"])
lb2 = comps(rest); sz2 = np.bincount(lb2[lb2 >= 0])
main = np.argmax(sz2)
hair |= rest & (lb2 != main)
L[zone & hair] = ID["hair"]
L[zone & ~hair & (L != ID["neck"])] = ID["head"]
np.save("L.npy", L)
for i, n in enumerate(NAMES):
    print("%-13s %7d граней" % (n, (L == i + 1).sum()))
