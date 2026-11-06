import numpy as np, trimesh, os, sys
from scipy.ndimage import gaussian_filter, label as cclabel, distance_transform_edt as edt
from skimage.measure import marching_cubes
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
V = 0.35  # mm voxel
xs = np.arange(-32, 32, V); ys = np.arange(-24, 22, V); zs = np.arange(-1, 108, V)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

def ell(c, r):
    return ((X-c[0])/r[0])**2 + ((Y-c[1])/r[1])**2 + ((Z-c[2])/r[2])**2 <= 1
def capsule(a, b, r):
    a, b = np.array(a, float), np.array(b, float); ab = b - a
    P = np.stack([X-a[0], Y-a[1], Z-a[2]], -1)
    t = np.clip((P @ ab) / (ab @ ab), 0, 1)
    D = P - t[..., None]*ab
    return (D**2).sum(-1) <= r*r
def loft(z0, z1, a0, a1, b0, b1, n=4, yc=0.0, xc=0.0):
    t = np.clip((Z - z0)/(z1 - z0), 0, 1)
    a = a0 + (a1-a0)*t; b = b0 + (b1-b0)*t
    inside = (np.abs(X-xc)/a)**n + (np.abs(Y-yc)/b)**n <= 1
    return inside & (Z >= z0) & (Z <= z1)
def cone(base, tip, r, flat=0.55):
    base, tip = np.array(base, float), np.array(tip, float); ax = tip - base; L = np.linalg.norm(ax); ax /= L
    P = np.stack([X-base[0], Y-base[1], Z-base[2]], -1)
    h = P @ ax
    R = P - h[..., None]*ax
    R[..., 1] /= flat  # flatten ear front-to-back
    rr = np.sqrt((R**2).sum(-1))
    return (h >= -2) & (h <= L) & (rr <= r*(1 - np.clip(h, 0, L)/L)**0.9 + 0.3)

parts = {}
# --- boots (green): big rounded, flat sole, toe forward
for s, nm in ((-1, 'boot_L'), (1, 'boot_R')):
    b = ell((s*14.5, -3.0, 9), (13, 16, 23)) & (Z >= 0)
    parts[nm] = b
    parts['leg_' + nm[-1]] = capsule((s*5.8, 0, 22), (s*5.8, 0, 40), 4.0)
# shorts (light cyan), flared, small crotch notch
sh = loft(32, 52.5, 13.8, 11.6, 9.4, 8.6, n=3.5)
sh &= ~((np.abs(X) < 1.6) & (Z < 37) & (np.abs(Y) < 12))
parts['shorts'] = sh
parts['belt'] = loft(49.8, 53.8, 11.9, 11.6, 8.9, 8.7, n=4)
# shirt (green): inverted trapezoid, wide shoulders
shirt = loft(52.5, 73.5, 10.9, 20.0, 8.4, 9.2, n=3.2)
# V-neck front region
V_tip_z, V_top_z, V_half = 63.0, 74.5, 10.0
inV = (Z >= V_tip_z) & (np.abs(X) <= (Z - V_tip_z)/(V_top_z - V_tip_z)*V_half) & (Y < -1)
shirt_in = loft(52.5, 73.5, 10.9-0.9, 20.0-0.9, 8.4-0.9, 9.2-0.9, n=3.2)
parts['undershirt'] = shirt_in & inV
parts['shirt'] = shirt & ~inV
parts['neck'] = capsule((0, 0, 70), (0, 0, 80), 6.0)
# head (dark blue): bell / dome, widest near bottom, flat-ish base
head = ((np.abs(X)/22.5)**2.4 + (np.abs(Y)/15.5)**2.4 + (np.clip(Z-80.5,0,None)/18.5)**2.6 <= 1) & (Z >= 75.6)
parts['head'] = head
for s, nm in ((-1, 'ear_L'), (1, 'ear_R')):
    parts[nm] = cone((s*13.5, 0.5, 89.5), (s*21.5, 1.5, 106), 6.5)
# arms (blue) and gloves (dark blue mitts)
for s, k in ((-1, 'L'), (1, 'R')):
    parts['arm_' + k] = capsule((s*16.5, 0, 70), (s*19.0, -2.5, 49), 3.4)
    g = ell((s*19.5, -4.0, 43.5), (9.6, 8.2, 9.2))
    g |= ell((s*11.8, -7.5, 46.5), (3.6, 3.3, 4.4))   # thumb on inner side
    parts['glove_' + k] = g

# paint order: later = on top (wins shared voxels)
order = ['leg_L','leg_R','arm_L','arm_R','neck','ear_L','ear_R','shorts','belt','undershirt','shirt',
         'boot_L','boot_R','glove_L','glove_R','head']
lab = np.zeros(X.shape, np.int16)
for i, n in enumerate(order, 1):
    lab[parts[n]] = i
np.savez_compressed(os.path.join(OUT, 'labels.npz'), lab=lab, order=np.array(order), V=V, origin=np.array([xs[0], ys[0], zs[0]]))

colors = {'boot':(122,248,82),'leg':(80,160,250),'shorts':(120,246,250),'belt':(110,240,190),
          'undershirt':(160,250,252),'shirt':(122,248,82),'neck':(120,40,230),'head':(10,10,240),
          'ear':(122,248,82),'arm':(40,90,245),'glove':(10,10,240)}
def mesh_mask(m):
    idx = np.argwhere(m); lo = np.maximum(idx.min(0)-6, 0); hi = idx.max(0)+7
    sub = np.pad(m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 3)
    d = edt(~sub) - edt(sub)
    d = gaussian_filter(d.astype(np.float32), 1.4)
    v, fc, _, _ = marching_cubes(d, 0.0, spacing=(V, V, V))
    v += np.array([xs[lo[0]], ys[lo[1]], zs[lo[2]]]) - 3*V
    return v, fc
scene = trimesh.Scene(); full = []
for i, n in enumerate(order, 1):
    m = (lab == i)
    cc, k = cclabel(m)
    if k > 1:  # keep largest piece
        sizes = np.bincount(cc.ravel())[1:]; m = cc == (np.argmax(sizes)+1)
    v, fc = mesh_mask(m)
    mesh = trimesh.Trimesh(v, fc[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=15)
    col = colors[n.split('_')[0]]
    mesh.visual.vertex_colors = np.tile(np.array(col + (255,), np.uint8), (len(mesh.vertices), 1))
    mesh.export(os.path.join(OUT, f'{n}.stl'))
    scene.add_geometry(mesh, node_name=n, geom_name=n)
    print(n, len(mesh.faces), 'watertight' if mesh.is_watertight else 'OPEN')
scene.export(os.path.join(OUT, 'figure.glb'))
# one solid piece too
v, fc = mesh_mask(lab > 0)
whole = trimesh.Trimesh(v, fc[:, ::-1], process=True); trimesh.smoothing.filter_taubin(whole, iterations=8)
whole.export(os.path.join(OUT, 'figure_whole.stl')); print('whole', whole.bounds.round(1).tolist(), whole.is_watertight)
