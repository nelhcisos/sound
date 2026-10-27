import trimesh, numpy as np, glob, os, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
D=sys.argv[1]; OUT=sys.argv[2]
COL=dict(eyes=(0.03,0.03,0.03),tie=(0.92,0.72,0.2),hair=(0.45,0.3,0.18),glasses=(0.2,0.2,0.22),head=(0.98,0.82,0.68),jacket=(0.25,0.25,0.32),
 shirt=(0.92,0.92,0.95),arm_right=(0.3,0.3,0.4),hand=(0.98,0.82,0.68),mug=(0.95,0.65,0.2),
 pants=(0.2,0.25,0.4),shoe_left=(0.12,0.12,0.12),shoe_right=(0.12,0.12,0.12),
 platform=(0.9,0.9,0.95),pant_left=(0.22,0.22,0.27),pant_right=(0.28,0.28,0.34),body=(0.16,0.16,0.2),
 sleeve_left=(0.3,0.3,0.36),sleeve_right=(0.34,0.34,0.4),hands=(0.98,0.82,0.68),neck=(0.95,0.78,0.64))
import os
if os.environ.get('HAIR'): COL['hair']=tuple(float(x) for x in os.environ['HAIR'].split(','))
parts={}
for f in sorted(glob.glob(D+'/*.stl')):
    n=os.path.basename(f)[:-4]; m=trimesh.load(f)
    if len(m.faces)>12000: m=m.simplify_quadric_decimation(face_count=12000)
    parts[n]=m
allv=np.vstack([m.vertices for m in parts.values()]); c=allv.mean(0)
L=np.array([0.6,-0.3,0.75]); L/=np.linalg.norm(L)
fig=plt.figure(figsize=(12,7.5),dpi=110,facecolor='white')
for k,(ex,title) in enumerate([(0,'в сборе'),(0.9,'в разборе')]):
    ax=fig.add_axes([k*0.5,0,0.5,0.93],projection='3d')
    tris=[];cols=[]
    for n,m in parts.items():
        d=m.centroid-c; d[2]*=1.3
        off=d*ex
        if n=='glasses': off=off+np.array([6,0,0])*ex
        if n in ('shirt','tie','eyes'): off=off+np.array([5 if n!='tie' else 8,0,0])*ex
        if n in('hand','mug'): off=off+np.array([4,-2,0])*ex
        t=m.triangles+off; tris.append(t)
        sh=np.clip(m.face_normals@L,0,1)*0.65+0.35
        cols.append(np.outer(sh,COL.get(n,(.6,.6,.6))))
    tris=np.vstack(tris); cols=np.clip(np.vstack(cols),0,1)
    ax.add_collection3d(Poly3DCollection(tris,facecolors=cols,edgecolor='none'))
    lo,hi=tris.reshape(-1,3).min(0),tris.reshape(-1,3).max(0); cc=(lo+hi)/2; r=(hi-lo).max()/2*0.75
    ax.set_xlim(cc[0]-r,cc[0]+r);ax.set_ylim(cc[1]-r,cc[1]+r);ax.set_zlim(cc[2]-r,cc[2]+r)
    ax.set_box_aspect((1,1,1)); ax.view_init(12,-35); ax.axis('off'); ax.set_title(title,y=0.97,fontsize=14)
fig.suptitle('%d деталей, высота 40 мм'%len(parts),y=0.99,fontsize=15)
plt.savefig(OUT,facecolor='white')
