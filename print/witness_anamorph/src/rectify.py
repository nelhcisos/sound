import cv2, numpy as np
from scipy import ndimage as ndi
ASP=1.28; FW=2048; FH=int(round(FW/ASP))
C=np.load('face2_corners.npy').astype(np.float32)
dst=np.float32([[0,0],[FW,0],[FW,FH],[0,FH]])
R2=cv2.getPerspectiveTransform(C,dst); np.save('R2.npy',R2)
im=cv2.imread('screens/i2.png')
rect=cv2.warpPerspective(im,R2,(FW,FH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite('face2_rect.png',rect)
# фигура в координатах грани
H7=np.load('H_7to2.npy'); Hf=R2@H7; np.save('H_7toface.npy',Hf)
f=np.load('fig_mask.npz'); ap=f['ap'].astype(np.uint8)*255
S=cv2.warpPerspective(ap,Hf,(FW,FH),flags=cv2.INTER_LINEAR)>127
np.save('S_face.npy',S)
ys,xs=np.where(S); print('S bbox u',xs.min()/FW,xs.max()/FW,'v',ys.min()/FH,ys.max()/FH)
# аффинная часть H_7->face в центре фигуры: насколько вид из игры отличается от фронтального
cx,cy=426,444
def J(H,x,y):
    e=1.0; p=lambda a,b: cv2.perspectiveTransform(np.float32([[[a,b]]]),H)[0,0]
    return np.c_[(p(x+e,y)-p(x-e,y))/2/e,(p(x,y+e)-p(x,y-e))/2/e]
Jm=J(Hf,cx,cy); U,s,Vt=np.linalg.svd(Jm); print('J',Jm.round(3),'sv',s.round(3),'anisotropy',round(s[0]/s[1],3),'rot',round(np.degrees(np.arctan2(Jm[1,0],Jm[0,0])),1))
