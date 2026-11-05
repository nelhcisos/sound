# render2.py dir out.png up(z|y|-y) [explode=0,0.8] [views=front,left,back,right] [exclude regex] [frontsign]
import trimesh, numpy as np, glob, os, sys, re
from PIL import Image, ImageDraw
D,OUT,UP=sys.argv[1],sys.argv[2],sys.argv[3]
EX=[float(x) for x in (sys.argv[4] if len(sys.argv)>4 else '0,0.8').split(',')]
VIEWS=(sys.argv[5] if len(sys.argv)>5 else 'front,left,back,right').split(',')
EXC=sys.argv[6] if len(sys.argv)>6 and sys.argv[6] else None
FS=float(sys.argv[7]) if len(sys.argv)>7 else 1.0
W=int(os.environ.get('W','480'))
UPM={'z':np.eye(3),'y':np.array([[1,0,0],[0,0,-1],[0,1,0]]),'-y':np.array([[1,0,0],[0,0,1],[0,-1,0]])}[UP]
pal=[(230,75,60),(60,140,230),(250,200,40),(80,190,110),(170,90,210),(250,140,40),(40,190,200),(230,110,170),(140,110,80),(120,120,130),(200,230,90),(90,90,200),(250,160,140),(60,110,70),(200,60,120),(160,200,240),(240,220,170),(100,60,40),(0,150,150),(180,180,0),(255,90,0),(0,90,180),(150,0,90),(90,150,0)]
parts=[]
for f in sorted(glob.glob(D+'/*.stl')):
    n=os.path.basename(f)[:-4]
    if EXC and re.search(EXC,n): continue
    m=trimesh.load(f,force='mesh'); m.vertices=m.vertices@UPM.T
    k=int(max(80000,min(600000,m.area*3)))
    p,fi=trimesh.sample.sample_surface(m,k); parts.append((n,p,m.face_normals[fi],m.centroid)); del m
allp=np.vstack([p for _,p,_,_ in parts]); C=(allp.min(0)+allp.max(0))/2; C[2]=allp[:,2].min()
F=np.diag([FS,FS,1])
ROT={'front':np.eye(3),'left':np.array([[0,1,0],[-1,0,0],[0,0,1]]),'back':np.array([[-1,0,0],[0,-1,0],[0,0,1]]),'right':np.array([[0,-1,0],[1,0,0],[0,0,1]])}
L=np.array([0.4,-0.7,0.6]); L/=np.linalg.norm(L)
tiles=[]
for ex in EX:
  for v in VIEWS:
    R=ROT[v]@F; pts=[];cols=[]
    for i,(n,p,nr,c) in enumerate(parts):
        off=(c-C)*np.array([1,1,0.7])*ex
        q=(p+off-C)@R.T; nn=nr@R.T
        sh=np.clip(nn@L,0,1)*0.65+0.35
        pts.append(q); cols.append(np.outer(sh,pal[i%len(pal)]))
    tiles.append((np.vstack(pts),np.vstack(cols),v,ex))
ext=max(max(np.ptp(t[0][:,0]),t[0][:,2].max()) for t in tiles)
S=(W-20)/ext
imgs=[]
for q,col,v,ex in tiles:
    H=W; x=(q[:,0]*S+W/2).astype(int); y=(H-10-q[:,2]*S).astype(int)
    ok=(x>=0)&(x<W)&(y>=0)&(y<H); x,y,d,cc=x[ok],y[ok],q[ok,1],col[ok]
    o=np.lexsort((d,y*W+x)); key=(y*W+x)[o]; first=np.r_[True,key[1:]!=key[:-1]]
    img=np.full((H*W,3),255,np.uint8); img[key[first]]=np.clip(cc[o][first],0,255).astype(np.uint8)
    im=Image.fromarray(img.reshape(H,W,3)); ImageDraw.Draw(im).text((8,8),'%s ex=%.1f'%(v,ex),fill=(0,0,0)); imgs.append(im)
cn=len(VIEWS); rows=len(EX)
out=Image.new('RGB',(W*cn+170,max(W*rows,22*len(parts)+20)),'white')
for k,im in enumerate(imgs): out.paste(im,((k%cn)*W,(k//cn)*W))
dr=ImageDraw.Draw(out)
for i,(n,_,_,_) in enumerate(parts):
    dr.rectangle([W*cn+8,10+i*22,W*cn+26,26+i*22],fill=pal[i%len(pal)]); dr.text((W*cn+32,12+i*22),n[:22],fill=(0,0,0))
out.save(OUT,quality=88); print('ok',OUT,out.size)
