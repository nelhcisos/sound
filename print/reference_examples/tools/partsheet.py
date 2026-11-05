# partsheet.py dir out.jpg up [exclude] [views=front,back,top,bottom] ; T env = tile px
import trimesh, numpy as np, glob, os, sys, re
from PIL import Image, ImageDraw
D,OUT,UP=sys.argv[1],sys.argv[2],sys.argv[3]
EXC=sys.argv[4] if len(sys.argv)>4 and sys.argv[4] else None
VIEWS=(sys.argv[5] if len(sys.argv)>5 else 'front,back,top,bottom').split(',')
T=int(os.environ.get('T','260')); NCOL=int(os.environ.get('NCOL','2'))
ONLY=os.environ.get('ONLY')
UPM={'z':np.eye(3),'y':np.array([[1,0,0],[0,0,-1],[0,1,0]]),'-y':np.array([[1,0,0],[0,0,1],[0,-1,0]])}[UP]
# view rotation: maps world to camera (x right, y depth into screen, z up)
ROT={'front':np.eye(3),'back':np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
     'left':np.array([[0,1,0],[-1,0,0],[0,0,1]]),'right':np.array([[0,-1,0],[1,0,0],[0,0,1]]),
     'top':np.array([[1,0,0],[0,0,-1],[0,1,0]]),'bottom':np.array([[1,0,0],[0,0,1],[0,-1,0]])}
Lv=np.array([0.35,-0.8,0.5]); Lv/=np.linalg.norm(Lv)
fs=sorted(glob.glob(D+'/*.stl'))
fs=[f for f in fs if not (EXC and re.search(EXC,os.path.basename(f)))]
if ONLY: fs=[f for f in fs if re.search(ONLY,os.path.basename(f))]
tiles=[]
for f in fs:
    m=trimesh.load(f,force='mesh'); m.vertices=m.vertices@UPM.T
    p,fi=trimesh.sample.sample_surface(m,int(os.environ.get('NS','700000'))); nr=m.face_normals[fi]
    row=[]
    for v in VIEWS:
        R=ROT[v]; q=(p-m.centroid)@R.T; nn=nr@R.T
        sh=np.clip(nn@Lv,0,1)*0.75+0.25
        # flat faces facing camera highlighted: |n.view|>0.995 → tint
        flat=nn[:,1]<-0.995
        col=np.outer(sh,[200,200,210]); col[flat]=np.outer(sh[flat],[255,120,90])
        e=max(np.ptp(q[:,0]),np.ptp(q[:,2]))*1.05; s=(T-6)/e
        x=((q[:,0]-(q[:,0].max()+q[:,0].min())/2)*s+T/2).astype(int); y=(T/2-(q[:,2]-(q[:,2].max()+q[:,2].min())/2)*s).astype(int)
        ok=(x>=0)&(x<T)&(y>=0)&(y<T); x,y,d,cc=x[ok],y[ok],q[ok,1],col[ok]
        o=np.lexsort((d,y*T+x)); key=(y*T+x)[o]; first=np.r_[True,key[1:]!=key[:-1]]
        img=np.full((T*T,3),255,np.uint8); img[key[first]]=np.clip(cc[o][first],0,255).astype(np.uint8)
        im=Image.fromarray(img.reshape(T,T,3)); ImageDraw.Draw(im).text((4,4),v,fill=(0,0,0)); row.append(im)
    ext=m.extents
    tiles.append((os.path.basename(f)[:-4],row,ext))
    del m
nv=len(VIEWS); cw=T*nv; rh=T+16
nrow=(len(tiles)+NCOL-1)//NCOL
out=Image.new('RGB',(cw*NCOL+(NCOL-1)*10,rh*nrow),'white'); dr=ImageDraw.Draw(out)
for i,(n,row,ext) in enumerate(tiles):
    cx=(i%NCOL)*(cw+10); cy=(i//NCOL)*rh
    dr.text((cx+4,cy+2),'%s  %.0fx%.0fx%.0f mm'%(n,*ext),fill=(0,0,160))
    for k,im in enumerate(row): out.paste(im,(cx+k*T,cy+16))
out.save(OUT,quality=85); print('ok',OUT,out.size)
