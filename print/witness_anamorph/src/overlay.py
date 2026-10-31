import cv2, numpy as np
f=np.load('fig_mask.npz'); ap=f['ap'].astype(np.uint8)
q=cv2.imread('screens/i7.png',0); t=cv2.imread('screens/i2.png')
cl=cv2.createCLAHE(3.0,(8,8))
prep=lambda g: cl.apply(cv2.GaussianBlur(g,(0,0),1.0))
sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=0.02)
qm=cv2.dilate(ap,np.ones((15,15)))*255
kq,dq=sift.detectAndCompute(prep(q),qm)
sc=0.5; tt=cv2.resize(cv2.cvtColor(t,cv2.COLOR_BGR2GRAY),None,fx=sc,fy=sc)
kt,dt=sift.detectAndCompute(prep(tt),None)
m=cv2.BFMatcher().knnMatch(dq,dt,k=2); good=[a for a,b in m if a.distance<0.8*b.distance]
src=np.float32([kq[g.queryIdx].pt for g in good]); dst=np.float32([kt[g.trainIdx].pt for g in good])/sc
H,inl=cv2.findHomography(src,dst,cv2.RANSAC,6.0)
print(int(inl.sum()), H)
np.save('H_7to2.npy',H)
cs,_=cv2.findContours(ap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
o=t.copy()
for c in cs:
    p=cv2.perspectiveTransform(c.astype(np.float32),H).astype(np.int32)
    cv2.polylines(o,[p],True,(0,0,255),3)
for (x,y),ok in zip(dst,inl.ravel()):
    if ok: cv2.circle(o,(int(x),int(y)),4,(255,0,0),-1)
cv2.imwrite('ov2.png',cv2.resize(o,None,fx=0.55,fy=0.55))
# warped figure image into i2 frame for side-by-side
w=cv2.warpPerspective(cv2.imread('screens/i7.png'),H,(t.shape[1],t.shape[0]))
x0,y0,x1,y1=600,150,1500,1150
side=np.hstack([t[y0:y1,x0:x1],w[y0:y1,x0:x1]])
cv2.imwrite('side2.png',cv2.resize(side,None,fx=0.5,fy=0.5))
