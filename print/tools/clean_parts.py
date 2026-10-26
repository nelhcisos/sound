import trimesh, fast_simplification as fs, numpy as np, glob, os, sys
for f in sorted(glob.glob(sys.argv[1]+'/*.stl')):
    m=trimesh.load(f); n=os.path.basename(f)[:-4]
    bodies=[b for b in m.split(only_watertight=False) if b.volume>1.0 or n=='eyes']
    if n!='eyes': bodies=[max(bodies,key=lambda b:b.volume)]
    m=trimesh.util.concatenate(bodies); n0=len(m.faces)
    if n0>40000:
        for red in (0.75,0.6,0.5,0.3):
            v,fc=fs.simplify(m.vertices.astype(np.float32),m.faces.astype(np.int32),target_reduction=red,agg=3)
            t=trimesh.Trimesh(v,fc,process=True); trimesh.repair.fix_normals(t)
            if t.is_watertight and len(t.split(only_watertight=False))==len(bodies): m=t; break
    m.export(f)
    print('%-11s тел=%d граней %7d->%7d замкнута=%s %.0f мм³ %.1fМБ'%(n,len(bodies),n0,len(m.faces),m.is_watertight,m.volume,os.path.getsize(f)/1e6))
