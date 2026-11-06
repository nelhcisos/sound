from PIL import Image; import numpy as np
from scipy import ndimage as nd
im=np.array(Image.open('sk.jpg').convert('RGB')).astype(float)[:, :615]
ref={'white':(245,245,245),'green':(117,251,76),'dblue':(0,0,246),'purple':(52,7,246),'vee':(164,252,254),
     'belt':(117,251,178),'shorts':(117,251,254),'arm':(34,83,244),'leg':(76,167,248)}
names=list(ref); R=np.array([ref[k] for k in names],float)
d=((im[:,:,None,:]-R[None,None])**2).sum(-1); cls=d.argmin(-1)
parts={}
def comps(mask,minsize=800):
    lab,n=nd.label(mask); out=[]
    for i in range(1,n+1):
        m=lab==i
        if m.sum()>=minsize: out.append(m)
    return out
for k in names[1:]:
    m=cls==names.index(k)
    m=nd.binary_opening(m,iterations=2); m=nd.binary_closing(m,iterations=3)
    for c in comps(m):
        c=nd.binary_fill_holes(c)
        ys,xs=np.nonzero(c); cy,cx=ys.mean(),xs.mean()
        if k=='green': nm='ear' if cy<250 else ('shirt' if cy<600 else 'boot')
        elif k=='dblue': nm='head' if cy<400 else 'glove'
        else: nm=k
        if nm in('ear','boot','glove','arm','leg'): nm+='_L' if cx<330 else '_R'   # image-left
        if nm in parts: parts[nm]|=c
        else: parts[nm]=c
for k,m in parts.items():
    ys,xs=np.nonzero(m); print(f'{k:8s} y {ys.min()}-{ys.max()} x {xs.min()}-{xs.max()} px {m.sum()}')
np.savez_compressed('front_parts.npz',**parts)
vis=np.full(im.shape,255,np.uint8); rng=np.random.default_rng(3)
for k,m in parts.items(): vis[m]=rng.integers(40,230,3)
Image.fromarray(vis).resize((307,522)).save('front_seg.png')
