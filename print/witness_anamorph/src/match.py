import cv2, numpy as np
f=np.load('fig_mask.npz'); ap=f['ap']
q=cv2.imread('screens/i7.png',0)
cl=cv2.createCLAHE(3.0,(8,8))
def prep(g):
    g=cv2.GaussianBlur(g,(0,0),1.0); return cl.apply(g)
qm=(cv2.dilate(ap.astype(np.uint8),np.ones((15,15)))*255)
sift=cv2.SIFT_create(nfeatures=4000,contrastThreshold=0.02)
kq,dq=sift.detectAndCompute(prep(q),qm)
print('query kp',len(kq))
for i in (1,2,3):
    t=cv2.imread(f'screens/i{i}.png',0)
    for sc in (0.5,0.75,1.0,1.5):
        tt=cv2.resize(t,None,fx=sc,fy=sc)
        kt,dt=sift.detectAndCompute(prep(tt),None)
        m=cv2.BFMatcher().knnMatch(dq,dt,k=2)
        good=[a for a,b in m if a.distance<0.8*b.distance]
        if len(good)<8: print(i,sc,'few',len(good)); continue
        src=np.float32([kq[g.queryIdx].pt for g in good]); dst=np.float32([kt[g.trainIdx].pt for g in good])/sc
        H,inl=cv2.findHomography(src,dst,cv2.RANSAC,8.0)
        n=int(inl.sum()) if inl is not None else 0
        print(i,sc,'good',len(good),'inliers',n, None if H is None else np.round(H,3).tolist() if n>=10 else '')
