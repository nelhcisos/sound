# joint.py A.stl B.stl out.png label : anatomy of a planar cut + key joint
import trimesh, numpy as np, sys, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
A,B,OUT,LAB=sys.argv[1:5]
ma=trimesh.load(A,force='mesh'); mb=trimesh.load(B,force='mesh')
def samp(m,n=1500000):
    p,fi=trimesh.sample.sample_surface(m,int(min(n,max(2e5,m.area*40)))); return p,m.face_normals[fi]
pa,na=samp(ma); pb,nb=samp(mb)
tb=cKDTree(pb); ta=cKDTree(pa)
da,ia=tb.query(pa,distance_upper_bound=0.8); ca=da<0.8
db,ib=ta.query(pb,distance_upper_bound=0.8); cb=db<0.8
P=pa[ca]; N=na[ca]
# dominant normal among contact points of A (cut face normal points from A to B = outward normal of A)
best=None
for k in range(400):
    i=np.random.randint(len(N)); n=N[i]
    c=(N@n>0.985).sum()
    if best is None or c>best[0]: best=(c,n)
n=best[1]; sel=N@n>0.985
n=N[sel].mean(0); n/=np.linalg.norm(n); sel=N@n>0.985
off=np.median(P[sel]@n)
flat=P[sel]; flat=flat[np.abs(flat@n-off)<0.4]
u=np.cross(n,[0,0,1] if abs(n[2])<0.9 else [1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
# in-plane PCA of the flat face
q=np.c_[(flat-flat.mean(0))@u,(flat-flat.mean(0))@v]; w,V=np.linalg.eigh(np.cov(q.T)); ax=V[:,::-1]
U=ax[0,0]*u+ax[1,0]*v; Vv=ax[0,1]*u+ax[1,1]*v
def ext(pts,c): x=(pts-c)@U; y=(pts-c)@Vv; return np.percentile(x,98)-np.percentile(x,2), np.percentile(y,98)-np.percentile(y,2)
c0=flat.mean(0)
fx,fy=ext(flat,c0)
# cut outline span: all contact points close to plane
res={'flat_face':(fx,fy),'flat_pts':len(flat)}
# tenon of A into B: A contact points beyond the plane (toward B)
sA=(P@n-off)
ten=P[sA>0.4]; ten_dir='A->B'
# mortise of A (B's tenon into A): A contact points below plane by >0.4 with normal not parallel
mor=P[(sA<-0.4)]
out=[]
for name,T,sgn in (('tenon A into B',ten,1),('tenon B into A',mor,-1)):
    if len(T)<200: continue
    # cluster: keep largest blob
    from scipy import ndimage
    g=np.floor((T-T.min(0))/0.5).astype(int); sh=g.max(0)+1
    occ=np.zeros(sh,bool); occ[tuple(g.T)]=1; lab,nl=ndimage.label(occ,np.ones((3,3,3)))
    L=lab[tuple(g.T)]; big=np.bincount(L)[1:].argmax()+1; T=T[L==big]
    d=np.abs(T@n-off)
    tc=T.mean(0); tx=(T-tc)@U; ty=(T-tc)@Vv
    # side walls: points whose normal ⟂ n
    out.append((name,np.percentile(tx,99)-np.percentile(tx,1),np.percentile(ty,99)-np.percentile(ty,1),d.max(),tc))
gap=np.percentile(da[ca],[10,50,90])
gflat=np.median(da[ca][sel][np.abs(P[sel]@n-off)<0.4]) if sel.sum() else -1
print('%s | cut face %.1f x %.1f mm | gap all p10/50/90 %.2f/%.2f/%.2f, on flat face %.2f'%(LAB,fx,fy,*gap,gflat))
for o in out: print('   %s: %.1f x %.1f mm, depth %.1f mm'%o[:4])
# section through tenon center (or flat center), plane containing n and U
cc=out[0][4] if out else c0
fig,axs=plt.subplots(1,2,figsize=(10,5),dpi=110)
for axi,(pn,a1,a2,tt) in enumerate([(Vv,U,n,'side cut (⊥ to joint)'),(n,U,Vv,'cut parallel to joint, mid-key')]):
    o=cc if axi==0 else (cc if not out else cc)
    for m,col,lab in ((ma,'#e0604a',os.path.basename(A)[:-4]),(mb,'#4a8be0',os.path.basename(B)[:-4])):
        s=m.section(plane_origin=o,plane_normal=pn)
        if s is None: continue
        for e in s.discrete:
            x=(e-o)@a1; y=(e-o)@a2
            axs[axi].fill(x,y,color=col,alpha=0.55,lw=0); axs[axi].plot(x,y,color=col,lw=0.6)
        axs[axi].plot([],[],color=col,lw=6,label=lab)
    R=max(fx,fy)*0.75
    axs[axi].set_xlim(-R,R); axs[axi].set_ylim(-R,R); axs[axi].set_aspect('equal'); axs[axi].grid(alpha=0.3); axs[axi].set_title(tt,fontsize=9); axs[axi].legend(fontsize=7)
fig.suptitle(LAB,fontsize=10); plt.tight_layout(); plt.savefig(OUT)
