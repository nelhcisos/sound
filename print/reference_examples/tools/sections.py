# sections.py dir out.png  "x=0" "z=30" "y=-40" ...  : cross-sections of assembly, each part colored
import trimesh, numpy as np, glob, os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
D,OUT=sys.argv[1],sys.argv[2]; CUTS=sys.argv[3:]
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
TR={'头发B':'hair_back','头发F':'hair_front','尾巴':'tail','屁股装饰':'bow','底座':'base','手L':'hand_L','手R':'hand_R','眼镜':'glasses','脚L':'boot_L','脚R':'boot_R','脸':'face','腿':'legs','衣服L':'jacket_L','衣服R':'jacket_R','衬衣':'shirt','裙子':'skirt'}
def nm(f):
    b=os.path.basename(f)[:-4]
    for k,v in TR.items():
        if b.endswith(k): return v
    return b
pal=[(230,75,60),(60,140,230),(250,200,40),(80,190,110),(170,90,210),(250,140,40),(40,190,200),(230,110,170),(140,110,80),(120,120,130),(200,230,90),(90,90,200),(250,160,140),(60,110,70),(200,60,120),(160,200,240),(240,220,170)]
M=[(nm(f),trimesh.load(f,force='mesh')) for f in sorted(glob.glob(D+'/*.stl'))]
fig,axs=plt.subplots(1,len(CUTS),figsize=(7*len(CUTS),8),dpi=110)
axs=np.atleast_1d(axs)
for ax,cut in zip(axs,CUTS):
    a,v=cut.split('='); v=float(v); k='xyz'.index(a)
    nrm=np.zeros(3); nrm[k]=1; o=nrm*v
    # 2D axes
    ax2={0:(1,2),1:(0,2),2:(0,1)}[k]
    for i,(n,m) in enumerate(M):
        s=m.section(plane_origin=o,plane_normal=nrm)
        if s is None: continue
        col=np.array(pal[i%len(pal)])/255
        for ent in s.discrete:
            pts=ent[:,list(ax2)]
            ax.fill(pts[:,0],pts[:,1],color=col,alpha=0.85,lw=0)
            ax.plot(pts[:,0],pts[:,1],color='k',lw=0.3)
        ax.plot([],[],color=col,lw=6,label=n)
    ax.set_aspect('equal'); ax.set_title(cut); ax.legend(fontsize=7,loc='upper right'); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(OUT)
print('ok')
