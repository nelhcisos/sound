from PIL import Image;import numpy as np
from scipy import ndimage as ndi
a=np.asarray(Image.open('fig.png').convert('RGB')).astype(float)
r,g,b=a[...,0],a[...,1],a[...,2]; L=a.mean(2)
neutral=(np.abs(r-g)<35)&(np.abs(g-b)<45)
ap=(L>52)&neutral
ap=ndi.binary_opening(ap,iterations=1)
lab,n=ndi.label(ap); sz=ndi.sum(ap,lab,range(1,n+1))
keep=np.isin(lab,1+np.where(sz>3000)[0])
keep=ndi.binary_closing(keep,iterations=2)
keep=ndi.binary_fill_holes(keep)|keep
# fill only small holes
holes=ndi.binary_fill_holes(keep)&~keep
hl,hn=ndi.label(holes);hs=ndi.sum(holes,hl,range(1,hn+1))
keep|=np.isin(hl,1+np.where(hs<400)[0])
white=keep&(L>118)
white=ndi.binary_opening(white,iterations=1)
ys,xs=np.where(keep);print(xs.min(),xs.max(),ys.min(),ys.max(),n,sorted(sz)[-5:])
out=np.zeros(L.shape+(3,),np.uint8);out[keep]=(150,140,110);out[white]=(250,250,250)
Image.fromarray(out).save('fig_mask.png')
np.savez('fig_mask.npz',ap=keep,white=white)
