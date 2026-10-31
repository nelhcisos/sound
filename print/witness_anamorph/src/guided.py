import cv2, numpy as np, sys
def guided_quad(fn, P, band=18, out=None):
    """P: грубые углы TL,TR,BR,BL. Уточняет каждую сторону по максимуму градиента поперёк неё."""
    g=cv2.GaussianBlur(cv2.imread(fn,0).astype(float),(0,0),1.5)
    gx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    lines=[]
    for i in range(4):
        a,b=np.array(P[i],float),np.array(P[(i+1)%4],float)
        d=(b-a)/np.linalg.norm(b-a); n=np.array([-d[1],d[0]])
        pts=[]
        for t in np.linspace(0.08,0.92,120):
            c=a+(b-a)*t; best=None
            for s in np.arange(-band,band+0.5,0.5):
                q=c+n*s; x,y=int(round(q[0])),int(round(q[1]))
                if 0<=x<g.shape[1] and 0<=y<g.shape[0]:
                    v=abs(gx[y,x]*n[0]+gy[y,x]*n[1])
                    if best is None or v>best[0]: best=(v,q)
            pts.append(best[1])
        p=np.array(pts)
        for _ in range(4):
            m=p.mean(0); u,s_,vt=np.linalg.svd(p-m); dirv=vt[0]; nn=np.array([-dirv[1],dirv[0]])
            r=np.abs((p-m)@nn); k=r<max(1.2,np.percentile(r,60)); p=p[k]
        lines.append(np.cross(np.r_[m,1],np.r_[m+dirv,1]))
    C=[]
    for i in range(4):
        v=np.cross(lines[i-1],lines[i]); C.append(v[:2]/v[2])
    C=np.array(C)  # угол i = пересечение стороны i-1 и i -> TL,TR,BR,BL при обходе
    if out:
        im=cv2.imread(fn); cv2.polylines(im,[C.astype(np.int32)],True,(0,0,255),2); cv2.imwrite(out,im)
    return C
if __name__=='__main__':
    C3L=guided_quad('screens/i3.png',[(229,220),(771,100),(767,873),(224,853)],out='q3L.png')
    C3R=guided_quad('screens/i3.png',[(771,100),(1293,227),(1300,847),(767,873)],out='q3R.png')
    # порядок: TL,TR,BR,BL. Пересечения: угол0 = пересечение стороны3 (BL->TL) и стороны0 (TL->TR) и т.д.
    print('3L',C3L.round(1)); print('3R',C3R.round(1))
    np.save('face3L_corners.npy',C3L); np.save('face3R_corners.npy',C3R)
    im=cv2.imread('screens/i3.png'); 
    for C in (C3L,C3R): cv2.polylines(im,[C.astype(np.int32)],True,(0,0,255),2)
    cv2.imwrite('q3s.png',cv2.resize(im,None,fx=0.5,fy=0.5))
