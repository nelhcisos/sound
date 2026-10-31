import cv2, numpy as np
from scipy import ndimage as ndi
FW,FH=2048,1600
SRC={'f2':('screens/i2.png','face2_corners.npy'),'f1':('screens/i1.png','face1_corners.npy'),
     'f3L':('screens/i3.png','face3L_corners.npy'),'f3R':('screens/i3.png','face3R_corners.npy')}
dst=np.float32([[0,0],[FW,0],[FW,FH],[0,FH]])
out={}
for k,(fn,cf) in SRC.items():
    C=np.load(cf).astype(np.float32)
    Rm=cv2.getPerspectiveTransform(C,dst)
    rect=cv2.warpPerspective(cv2.imread(fn),Rm,(FW,FH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(f'rect_{k}.png',rect)
    lab=cv2.cvtColor(rect,cv2.COLOR_BGR2LAB).astype(float)
    Lc,bc=lab[...,0],lab[...,2]-128
    out[k]=(Lc,bc)
    print(k,'L pct',np.percentile(Lc,[5,25,50,75,95]).round(0),'b pct',np.percentile(bc,[5,25,50,75,95]).round(1))
np.savez('faces_lab.npz',**{k+'_L':v[0] for k,v in out.items()},**{k+'_b':v[1] for k,v in out.items()})
