"""Ровное дно платформы на мастере: досыпать воксели под платформой до её нижней точки. usage: flatten_base.py cutdir/labels.npz"""
import sys, numpy as np
d = dict(np.load(sys.argv[1])); lab = d["lab"]; names = [str(x) for x in d["names"]]; pid = names.index("platform") + 1
pl = lab == pid; foot = pl.any(2); zs = np.nonzero(pl.any((0, 1)))[0]; first = np.argmax(pl, axis=2); n = 0
for k in range(zs[0], zs[-1]):
    sel = foot & (first > k) & (lab[:, :, k] == 0); lab[:, :, k][sel] = pid; n += sel.sum()
print("досыпано %.0f мм³ мастера" % (n * float(d["pitch"]) ** 3))
d["lab"] = lab; np.savez_compressed(sys.argv[1], **d)
