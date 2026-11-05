import trimesh, numpy as np, glob, os, sys
D=sys.argv[1]
fs=sorted(glob.glob(D+'/*.stl'))
allb=[]
for f in fs:
    m=trimesh.load(f,force='mesh')
    b=m.bounds; allb.append(b)
    sp=m.split(only_watertight=False) if len(m.faces)<2e6 else None
    nb=len(sp) if sp is not None else -1
    vol=m.volume if m.is_watertight else float('nan')
    print('%-22s F=%8d wt=%d bodies=%3d size=%6.1f x %6.1f x %6.1f  min=(%6.1f %6.1f %6.1f) vol=%.0f cm3 area=%.0f'%(os.path.basename(f)[:-4][:22],len(m.faces),m.is_watertight,nb,*(b[1]-b[0]),*b[0],vol/1000 if vol==vol else -1,m.area))
allb=np.array(allb); lo=allb[:,0].min(0); hi=allb[:,1].max(0)
print('ASSEMBLY bounds', np.round(lo,1), np.round(hi,1), 'size', np.round(hi-lo,1))
