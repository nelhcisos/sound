import cv2, numpy as np
g=cv2.GaussianBlur(cv2.imread('screens/i2.png',0).astype(float),(0,0),1.5)
gx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
def fitx(xa,xb,ys,sign=None):
    pts=[]
    for y in ys:
        row=gx[y,xa:xb]; 
        x=np.argmax(np.abs(row)); pts.append((xa+x,y))
    p=np.array(pts,float); A=np.c_[p[:,1],np.ones(len(p))]
    for _ in range(3):
        c,*_=np.linalg.lstsq(A,p[:,0],rcond=None); r=np.abs(A@c-p[:,0]); k=r<max(2,np.percentile(r,70)); A,p=A[k],p[k]
    return c  # x = c0*y + c1
def fity(ya,yb,xs):
    pts=[]
    for x in xs:
        col=gy[ya:yb,x]; y=np.argmax(np.abs(col)); pts.append((x,ya+y))
    p=np.array(pts,float); A=np.c_[p[:,0],np.ones(len(p))]
    for _ in range(3):
        c,*_=np.linalg.lstsq(A,p[:,1],rcond=None); r=np.abs(A@c-p[:,1]); k=r<max(2,np.percentile(r,70)); A,p=A[k],p[k]
    return c  # y = c0*x + c1
L=fitx(150,330,range(250,1000,10)); R=fitx(1440,1620,range(250,1000,10))
T=fity(95,175,range(400,1400,10)); B=fity(1055,1125,range(400,1400,10))
print('L',L,'R',R,'T',T,'B',B)
def inter(cx,cy):  # x=a*y+b ; y=c*x+d
    a,b=cx; c,d=cy; y=(c*b+d)/(1-c*a); return np.array([a*y+b,y])
C=np.array([inter(L,T),inter(R,T),inter(R,B),inter(L,B)]); print(C.round(1))
np.save('face2_corners.npy',C)
im=cv2.imread('screens/i2.png')
cv2.polylines(im,[C.astype(np.int32)],True,(0,0,255),2)
cv2.imwrite('face2_quad.png',cv2.resize(im,None,fx=0.5,fy=0.5))
