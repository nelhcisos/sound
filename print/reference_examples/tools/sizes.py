import numpy as np, glob, os, sys
def ext(f):
    with open(f,'rb') as fh:
        fh.seek(80); n=np.frombuffer(fh.read(4),'<u4')[0]
        d=np.frombuffer(fh.read(n*50),dtype=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')]),count=n)
    v=d['v'].reshape(-1,3); return v.min(0), v.max(0)
for name,D,H,skip in [('Ubel',sys.argv[1],225,'FullBody|^Head$|Stand|Staff'),('Blair',sys.argv[2],140,'base|pumpkin|size|GK|face[2-5]'),
                      ('Frieren',sys.argv[3],140,'base|size|box|bag|staff|headGK'),('Victoria',sys.argv[4],230,'Base|muro|oscuridad'),
                      ('Mitsuri',sys.argv[5],280,'Base|Rope|Wind|Katana|Saya')]:
    import re; rows=[]
    for f in sorted(glob.glob(D+'/*.stl')):
        n=os.path.basename(f)[:-4]
        if re.search(skip,n): continue
        lo,hi=ext(f); e=np.sort(hi-lo); rows.append((e[0],e[1],e[2],n))
    rows.sort()
    print('%-9s %2d деталей персонажа; самые мелкие (мин. размер, %% высоты):'%(name,len(rows)), ', '.join('%s %.1f мм (%.1f%%)'%(r[3],r[0],100*r[0]/H) for r in rows[:4]))
