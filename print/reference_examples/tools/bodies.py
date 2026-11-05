# bodies.py file.stl out.png : list bodies, their big planar faces, render top + iso
import trimesh, numpy as np, sys, os
from PIL import Image, ImageDraw
f,OUT=sys.argv[1],sys.argv[2]
m=trimesh.load(f,force='mesh'); B=m.split(only_watertight=False)
B=sorted(B,key=lambda b:-b.volume if b.is_watertight else 0)
print(os.path.basename(f),len(B),'bodies')
for i,b in enumerate(B):
    # planar regions: facets
    fa=b.facets_area if len(b.facets) else []
    big=sorted([(a,b.facets_normal[k]) for k,a in enumerate(fa) if a>3],key=lambda z:-z[0])[:4]
    print(' %2d ext=%s vol=%.0f wt=%s lo=%s planes=%s'%(i,np.round(b.extents,1),b.volume if b.is_watertight else -1,b.is_watertight,np.round(b.bounds[0],0),[('%.0f'%a,tuple(np.round(n,1))) for a,n in big]))
# render top (z) and iso splat
pal=[(230,75,60),(60,140,230),(250,200,40),(80,190,110),(170,90,210),(250,140,40),(40,190,200),(230,110,170),(140,110,80),(120,120,130)]
W=900
def splat(R,w,h):
    pts=[];cols=[]
    L=np.array([0.3,-0.5,0.8]);L/=np.linalg.norm(L)
    for i,b in enumerate(B):
        p,fi=trimesh.sample.sample_surface(b,max(20000,int(b.area*8)))
        q=p@R.T; nn=b.face_normals[fi]@R.T; sh=np.clip(nn@L,0,1)*0.6+0.4
        pts.append(q); cols.append(np.outer(sh,pal[i%len(pal)]))
    q=np.vstack(pts);c=np.vstack(cols); lo=q.min(0);hi=q.max(0); S=min((w-20)/(hi[0]-lo[0]),(h-20)/(hi[1]-lo[1]))
    x=((q[:,0]-lo[0])*S+10).astype(int); y=(h-10-(q[:,1]-lo[1])*S).astype(int)
    key=y*w+x; o=np.lexsort((-q[:,2],key)); k=key[o]; fst=np.r_[True,k[1:]!=k[:-1]]
    img=np.full((h*w,3),255,np.uint8); img[k[fst]]=np.clip(c[o][fst],0,255).astype(np.uint8)
    return Image.fromarray(img.reshape(h,w,3))
top=splat(np.eye(3),W,500)
a=np.radians(55); Rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
iso=splat(Rx,W,500)
out=Image.new('RGB',(W,1000),'white'); out.paste(top,(0,0)); out.paste(iso,(0,500))
d=ImageDraw.Draw(out)
for i in range(min(len(B),10)): d.text((10,10+14*i),'%d'%i,fill=pal[i%len(pal)])
out.save(OUT)
