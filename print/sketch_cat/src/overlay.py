import numpy as np; from PIL import Image; from scipy import ndimage as nd
d=np.load('out_hi/labels.npz'); lab=d['lab']; order=list(d['order'])
col={'head':(0,0,246),'ear':(117,251,76),'neck':(52,7,246),'shirt':(117,251,76),'undershirt':(164,252,254),'belt':(117,251,178),
     'shorts':(117,251,254),'arm':(34,83,244),'glove':(0,0,246),'leg':(76,167,248),'boot':(117,251,76)}
C=np.zeros((len(order)+1,3),np.uint8)+255
for i,n in enumerate(order,1): C[i]=col[n.split('_')[0]]
sk=np.array(Image.open('sk.jpg').convert('RGB'))
DS=1; X0,Y0,Z0=60,630,40
def proj(axis,front_is_high):
    occ=lab>0; n=lab.shape[axis]
    idx=np.arange(n); shape=[1,1,1]; shape[axis]=n; idx=idx.reshape(shape)
    key=np.where(occ, idx if front_is_high else n-idx, -1)
    first=key.argmax(axis); lv=np.take_along_axis(lab,np.expand_dims(first,axis),axis).squeeze(axis)
    return lv, occ.any(axis)
# спереди: смотрим вдоль глубины (axis 1), передняя сторона = большие y
lvf,of=proj(1,True)          # (x,z)
lvs,os_=proj(0,False)        # сбоку: смотрим с левой стороны картинки... (y,z)
def paste(lv,oc,ox,oy,flipx=False):
    img=sk.copy()
    a=lv.T; o=oc.T
    a=np.kron(a,np.ones((DS,DS),int)); o=np.kron(o,np.ones((DS,DS),bool))
    edge=o^nd.binary_erosion(o,iterations=2)
    h,w=a.shape
    reg=img[oy:oy+h,ox:ox+w]
    reg[edge[:reg.shape[0],:reg.shape[1]]]=(255,0,0)
    return img
f=paste(lvf,of,X0,Z0)
s=paste(lvs,os_,Y0,Z0)
both=np.concatenate([f[:, :615],s[:, 615:]],1)
# цветная проекция модели
mf=np.full((1044,615,3),255,np.uint8); a=np.kron(lvf.T,np.ones((DS,DS),int)); mf[Z0:Z0+a.shape[0],X0:X0+a.shape[1]]=C[a]
ms=np.full((1044,356,3),255,np.uint8); b=np.kron(lvs.T,np.ones((DS,DS),int)); ms[Z0:Z0+b.shape[0],Y0-615:Y0-615+b.shape[1]]=C[b]
out=np.concatenate([both,np.full((1044,20,3),255,np.uint8),mf,ms],1)
Image.fromarray(out).resize((out.shape[1]//2,522)).save('overlay.png')
