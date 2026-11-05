import trimesh, numpy as np, sys
s=trimesh.load(sys.argv[1])
print(type(s))
geoms = s.dump() if hasattr(s,'dump') else [s]
out=sys.argv[2]
for i,g in enumerate(geoms):
    B=g.split(only_watertight=False)
    print(i, g.metadata.get('name'), 'tri',len(g.faces),'ext',np.round(g.extents,1),'wt',g.is_watertight,'vol',round(g.volume,0) if g.is_watertight else -1,'bodies',len(B),'lo',np.round(g.bounds[0],0))
    g.export('%s/part_%02d.stl'%(out,i))
