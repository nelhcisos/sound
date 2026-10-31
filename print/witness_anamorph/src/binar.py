import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
d=np.load('faces_lab.npz'); res={}
tiles=[]
for k in ('f2','f1','f3L','f3R'):
    L=d[k+'_L']; b=d[k+'_b']
    bs=ndi.gaussian_filter(b,2.0); Ls=ndi.gaussian_filter(L,2.0)
    # локальная нормировка b* (убирает перепады освещения)
    sm=lambda a,s: ndi.gaussian_filter(a,s)
    mu=sm(bs,60); sd=np.sqrt(np.maximum(sm((bs-mu)**2,60),1e-3))
    Lmu=sm(Ls,60); Lsd=np.sqrt(np.maximum(sm((Ls-Lmu)**2,60),1e-3))
    feat=(bs-mu)/sd - 0.6*(Ls-Lmu)/Lsd       # >0 -> олива (желтее и темнее)
    t=threshold_otsu(feat[50:-50,50:-50])
    white=feat<t
    res[k]=feat-t
    v=np.zeros(L.shape+(3,),np.uint8); v[white]=(235,235,235); v[~white]=(90,130,140)
    rect=cv2.imread(f'rect_{k}.png')
    tiles.append(np.hstack([cv2.resize(rect,(512,400)),cv2.resize(v,(512,400))]))
np.savez_compressed('faces_feat.npz',**res)
cv2.imwrite('bin_sheet.png',np.vstack(tiles))
