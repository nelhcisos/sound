import numpy as np, sys
from PIL import Image
lab=np.load(sys.argv[1])
C=np.array([[255,255,255],[120,80,40],[30,30,30],[245,205,170],[70,70,95],[235,235,240],[90,90,120],
 [245,205,170],[240,160,40],[0,0,0],[220,170,40],[60,80,130],[20,20,20],[20,20,20]],np.uint8)
s=lab>0; NX=lab.shape[0]
idx=NX-1-np.argmax(s[::-1],axis=0); has=s.any(0)
lb=np.take_along_axis(lab,idx[None],0)[0]*has
d=idx.astype(float); gy,gz=np.gradient(d)
sh=np.clip(0.8+0.12*gy-0.12*gz,0.45,1.1)
img=C[lb].astype(float)*sh[...,None]; img[~has]=255
img=np.clip(img,0,255).astype(np.uint8).transpose(1,0,2)[::-1]
k=int(sys.argv[3]) if len(sys.argv)>3 else 3
Image.fromarray(img).resize((img.shape[1]*k,img.shape[0]*k),Image.NEAREST).save(sys.argv[2])
