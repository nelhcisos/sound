import numpy as np; from PIL import Image; from scipy import ndimage as nd
A=np.array(Image.open('back.jpg').convert('RGB')).astype(int); print(A.shape)
r,g,b=A[...,0],A[...,1],A[...,2]
mag=(r>140)&(g<110)&(b>60)&(r-g>70)
bright=(A.sum(2)>600).mean(0)>0.5; cols=np.nonzero(bright)[0]; L,R=cols.min()+8,cols.max()-8; print('panel',L,R)
blk=(A.sum(2)<200); blk[:450]=False; blk[:, :L]=False; blk[:, R:]=False   # без шапки телефона
O=np.array(Image.open('sk.jpg').convert('RGB')).astype(int)[:, 620:950]; ob=O.sum(2)<250
ys,xs=np.nonzero(ob); print('orig side bbox',xs.min()+620,xs.max()+620,ys.min(),ys.max())
ys2,xs2=np.nonzero(blk); print('shot bbox',xs2.min(),xs2.max(),ys2.min(),ys2.max(), 'mag px',mag.sum())
# масштаб по росту ниже шапки: подбираем s,dx,dy перебором по дистанции до штрихов
D=nd.distance_transform_edt(~ob)
pts=np.c_[xs2,ys2][::7]
best=None
for s in np.arange(0.36,0.48,0.003):
    for dy in range(-60,61,3):
        # нижняя точка ботинка: совпадение по низу
        oy=ys.max()-ys2.max()*s+dy
        for dx in range(-60,61,3):
            ox=(xs.min()+xs.max())/2-(xs2.min()+xs2.max())/2*s+dx
            X=(pts[:,0]*s+ox).astype(int); Y=(pts[:,1]*s+oy).astype(int)
            ok=(X>=0)&(X<330)&(Y>=0)&(Y<1044)
            c=np.median(D[Y[ok],X[ok]])+ (1-ok.mean())*50
            if best is None or c<best[0]: best=(c,s,ox,oy)
print(best)
c,s,ox,oy=best
my,mx=np.nonzero(mag); np.save('mag_orig.npy',np.c_[mx*s+ox+620,my*s+oy])
img=np.array(Image.open('sk.jpg').convert('RGB'))
for x,y in [(x,y) for x,y in zip((mx*s+ox+620).astype(int),(my*s+oy).astype(int)) if 0<=x<971 and 0<=y<1044]: img[y,x]=(220,0,120)
Image.fromarray(img[30:1010,620:960]).save('mag_on_orig.png')
