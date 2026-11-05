# contacts.py dir [tol=0.6] : per-pair joint analysis for parts in assembled position
import trimesh, numpy as np, glob, os, sys
from scipy.spatial import cKDTree
from scipy import ndimage
sys.path.insert(0,os.path.dirname(__file__))
D=sys.argv[1]; TOL=float(sys.argv[2]) if len(sys.argv)>2 else 0.6
TR={'头发B':'hair_back','头发F':'hair_front','尾巴':'tail','屁股装饰':'bow','底座':'base','手L':'hand_L','手R':'hand_R','眼镜':'glasses','脚L':'boot_L','脚R':'boot_R','脸':'face','腿':'legs','衣服L':'jacket_L','衣服R':'jacket_R','衬衣':'shirt','裙子':'skirt'}
def nm(f):
    b=os.path.basename(f)[:-4]
    for k,v in TR.items():
        if b.endswith(k): return v
    return b
DENS=6.0
P={}
for f in sorted(glob.glob(D+'/*.stl')):
    m=trimesh.load(f,force='mesh'); n=int(min(1.5e6,max(5e4,m.area*DENS)))
    p,fi=trimesh.sample.sample_surface_even(m,n) if False else trimesh.sample.sample_surface(m,n)
    P[nm(f)]=dict(p=p,n=m.face_normals[fi],t=cKDTree(p),lo=m.bounds[0],hi=m.bounds[1],c=m.centroid,dens=n/m.area)
names=list(P)
rows=[]
for i,a in enumerate(names):
  for b in names[i+1:]:
    A,B=P[a],P[b]
    if np.any(A['lo']>B['hi']+TOL) or np.any(B['lo']>A['hi']+TOL): continue
    d,_=B['t'].query(A['p'],distance_upper_bound=TOL); mA=d<TOL
    if mA.sum()<30: continue
    d2,_=A['t'].query(B['p'],distance_upper_bound=TOL); mB=d2<TOL
    pa=A['p'][mA]; area=mA.sum()/A['dens']
    # plane fit
    c=pa.mean(0); u,s,vt=np.linalg.svd(pa-c,full_matrices=False); nrm=vt[2]
    if np.dot(B['c']-A['c'],nrm)<0: nrm=-nrm
    off=(pa-c)@nrm
    # robust: refit plane on points near median (main flat)
    med=np.median(off); core=np.abs(off-med)<0.4
    if core.sum()>10:
        c2=pa[core].mean(0); u,s,vt=np.linalg.svd(pa[core]-c2,full_matrices=False); n2=vt[2]
        if np.dot(n2,nrm)<0:n2=-n2
        nrm,c=n2,c2; off=(pa-c)@nrm
    flat=np.mean(np.abs(off)<0.3)
    # normals agreement of A contact faces with plane normal (flat face should have normal ~ +nrm)
    na=A['n'][mA]; nd=np.abs(na@nrm); facing=np.mean(nd>0.95)
    # gap stats
    gap=np.percentile(d[mA],[10,50,90])
    # features off-plane: cluster with 1mm voxels
    feat=[]
    fo=np.abs(off)>0.5
    if fo.sum()>20:
        q=pa[fo]; g=np.floor((q-q.min(0))/1.0).astype(int); sh=g.max(0)+1
        if np.prod(sh)<5e7:
            occ=np.zeros(sh,bool); occ[tuple(g.T)]=True; lab,nl=ndimage.label(occ,np.ones((3,3,3)))
            L=lab[tuple(g.T)]
            e1=vt[0];e2=vt[1]
            for k in range(1,nl+1):
                s_=q[L==k]
                if len(s_)<15: continue
                o=(s_-c)@nrm; x=(s_-c)@e1; y=(s_-c)@e2
                feat.append((len(s_)/A['dens'], o.min(), o.max(), np.ptp(x), np.ptp(y), (x.mean(),y.mean())))
    rows.append((a,b,area,flat,facing,gap,feat,nrm,c))
    print('\n%s <-> %s : contact %.0f mm2 (on %s), flat(<0.3mm)=%.0f%%, faces||plane=%.0f%%, gap p10/50/90=%.2f/%.2f/%.2f, normal=%s'%(a,b,area,a,flat*100,facing*100,*gap,np.round(nrm,2)))
    for fa in sorted(feat,key=lambda z:-z[0])[:6]:
        print('   feature area %.0f mm2 depth %+.1f..%+.1f mm (+ = toward %s) inplane %.1f x %.1f mm at (%.0f,%.0f)'%(fa[0],fa[1],fa[2],b,fa[3],fa[4],*fa[5]))
