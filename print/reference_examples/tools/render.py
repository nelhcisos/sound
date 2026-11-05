# splat renderer: render.py dir out.png [explode=0.0,0.8] [views=front,side,back]
import trimesh, numpy as np, glob, os, sys
from PIL import Image, ImageDraw, ImageFont
D,OUT=sys.argv[1],sys.argv[2]
EX=[float(x) for x in (sys.argv[3] if len(sys.argv)>3 else '0,0.8').split(',')]
VIEWS=(sys.argv[4] if len(sys.argv)>4 else 'front,side,back').split(',')
TR={'头发B':'hair_back','头发F':'hair_front','尾巴':'tail','屁股装饰':'bow','底座':'base','手L':'hand_L','手R':'hand_R','眼镜':'glasses','脚L':'boot_L','脚R':'boot_R','脸':'face','腿':'legs','衣服L':'jacket_L','衣服R':'jacket_R','衬衣':'shirt','裙子':'skirt'}
def nm(f):
    b=os.path.basename(f)[:-4]
    for k,v in TR.items():
        if b.endswith(k): return v
    return b
pal=[(230,75,60),(60,140,230),(250,200,40),(80,190,110),(170,90,210),(250,140,40),(40,190,200),(230,110,170),(140,110,80),(120,120,130),(200,230,90),(90,90,200),(250,160,140),(60,110,70),(200,60,120),(160,200,240),(240,220,170)]
parts=[]
for f in sorted(glob.glob(D+'/*.stl')):
    m=trimesh.load(f,force='mesh'); n=max(60000,min(400000,int(m.area*1.5)))
    p,fi=trimesh.sample.sample_surface(m,n); parts.append((nm(f),p,m.face_normals[fi],m.centroid))
allp=np.vstack([p for _,p,_,_ in parts]); C=(allp.min(0)+allp.max(0))/2; C[2]=allp[:,2].min()
ROT={'front':np.eye(3),'side':np.array([[0,1,0],[-1,0,0],[0,0,1]]),'back':np.array([[-1,0,0],[0,-1,0],[0,0,1]])}
L=np.array([0.4,-0.7,0.6]); L/=np.linalg.norm(L)
W=520; tiles=[]
for ex in EX:
  for v in VIEWS:
    R=ROT[v]; pts=[];cols=[]
    for i,(n,p,nr,c) in enumerate(parts):
        off=(c-C)*np.array([1,1,0.6])*ex
        q=(p+off-C)@R.T; nn=nr@R.T
        sh=np.clip(nn@L,0,1)*0.6+0.4
        pts.append(q); cols.append(np.outer(sh,pal[i%len(pal)]))
    q=np.vstack(pts); col=np.vstack(cols)
    tiles.append((q,col,v,ex))
# common scale
ext=max(np.ptp(np.vstack([t[0][:,[0,2]] for t in tiles]),0))
S=(W-20)/ext
imgs=[]
for q,col,v,ex in tiles:
    H=W; x=((q[:,0]-q[:,0].min()*0)*S+W/2).astype(int); y=(H-10-q[:,2]*S).astype(int)
    ok=(x>=0)&(x<W)&(y>=0)&(y<H); x,y,d,cc=x[ok],y[ok],q[ok,1],col[ok]
    o=np.lexsort((d,y*W+x)); key=(y*W+x)[o]; first=np.r_[True,key[1:]!=key[:-1]]
    img=np.full((H*W,3),255,np.uint8); img[key[first]]=np.clip(cc[o][first],0,255).astype(np.uint8)
    im=Image.fromarray(img.reshape(H,W,3)); ImageDraw.Draw(im).text((8,8),'%s ex=%.1f'%(v,ex),fill=(0,0,0)); imgs.append(im)
cols_n=len(VIEWS); rows=len(EX)
out=Image.new('RGB',(W*cols_n+200,W*rows),'white')
for k,im in enumerate(imgs): out.paste(im,((k%cols_n)*W,(k//cols_n)*W))
dr=ImageDraw.Draw(out)
for i,(n,_,_,_) in enumerate(parts):
    dr.rectangle([W*cols_n+10,10+i*22,W*cols_n+28,26+i*22],fill=pal[i%len(pal)]); dr.text((W*cols_n+34,12+i*22),n,fill=(0,0,0))
out.save(OUT); print('ok',OUT,out.size)
