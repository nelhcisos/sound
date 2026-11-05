import trimesh, numpy as np, sys, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
OUT=sys.argv[1]; fs=sys.argv[2:]
fig,axs=plt.subplots(1,len(fs),figsize=(4*len(fs),5),dpi=100)
for ax,f in zip(axs,fs):
    m=trimesh.load(f,force='mesh'); nsh=len(m.split(only_watertight=False))
    ths=[]
    for k,frac in enumerate([0.3,0.5,0.7]):
        o=m.bounds[0]+(m.bounds[1]-m.bounds[0])*frac
        s=m.section(plane_origin=o,plane_normal=[0,0,1])
        if s is None: continue
        loops=[e for e in s.discrete if len(e)>20]
        if len(loops)<2: continue
        # outer = largest bbox loop; others inside = cavity
        ar=[np.ptp(e[:,0])*np.ptp(e[:,1]) for e in loops]; oi=int(np.argmax(ar)); outer=loops[oi]
        t=cKDTree(outer[:,:2])
        for j,e in enumerate(loops):
            if j==oi: continue
            d,_=t.query(e[:,:2]); ths.append(np.percentile(d,10))
    s=m.section(plane_origin=m.bounds.mean(0),plane_normal=[1,0,0])
    if s is not None:
        for e in s.discrete: ax.fill(e[:,1],e[:,2],color='#e0604a',alpha=0.5); ax.plot(e[:,1],e[:,2],'k',lw=0.4)
    ax.set_aspect('equal'); ax.set_title('%s shells=%d'%(os.path.basename(f)[:-4],nsh),fontsize=8)
    print(os.path.basename(f),'shells',nsh,'min wall per cavity loop (p10):',np.round(ths,2))
plt.tight_layout(); plt.savefig(OUT)
