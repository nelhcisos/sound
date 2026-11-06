# Фигурка строго по скетчу: спереди — маска каждой детали (по цвету),
# сбоку — обведённый контур детали. Каждый горизонтальный срез детали —
# суперэллипс, вписанный ровно в ширину (спереди) и глубину (сбоку).
import numpy as np, os, sys, trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as nd
from skimage.measure import marching_cubes
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
F=dict(np.load('front_parts.npz'))
# мятная окантовка V-выреза попала в «пояс» — это часть рубашки
F['shirt']|=F['belt']&(np.arange(F['belt'].shape[0])[:,None]<500); F['belt']&=np.arange(F['belt'].shape[0])[:,None]>=500
F['neck']=F.pop('purple')
# скрытые «шипы» внутрь головы: шея вверх, уши вниз (снаружи их закрывает голова)
def grow(m,dy):
    out=m.copy()
    for k in range(1,abs(dy)+1): out|=np.roll(m,-k if dy<0 else k,0)
    return out
F['neck']=F['neck']|(grow(F['neck'],-30)&F['head'])
for e in ('ear_L','ear_R'): F[e]=F[e]|(grow(F[e],35)&F['head'])
F['neck']=F['neck']|(grow(F['neck'],25)&F['shirt'])
for l,b in (('leg_L','boot_L'),('leg_R','boot_R')): F[l]=F[l]|(grow(F[l],45)&F[b])|(grow(F[l],-15)&F['shorts'])
H,W=1044,971
SIDE={ # обводка вида сбоку, координаты пикселей скетча (x вправо = вперёд)
 'head':[(727,135),(750,127),(870,124),(885,135),(890,170),(885,210),(870,240),(845,262),(835,285),(820,298),(760,308),(700,318),(688,305),(686,270),(700,225),(715,180)],
 'ear':[(842,50),(852,80),(856,120),(858,180),(835,182),(805,172),(782,150),(790,120),(815,85)],
 'neck':[(735,305),(830,295),(835,330),(735,335)],
 'shirt':[(740,328),(850,325),(868,335),(875,380),(890,425),(905,465),(910,495),(905,512),(650,512),(660,490),(690,445),(715,400),(730,360)],
 'belt':[(650,495),(910,490),(912,540),(652,545)],
 'shorts':[(650,510),(908,505),(915,560),(910,615),(925,660),(915,690),(770,692),(748,680),(710,625),(690,600),(665,592),(655,560)],
 'arm':[(768,350),(812,342),(850,395),(880,445),(900,490),(875,510),(848,505),(825,455),(795,400)],
 'glove':[(745,640),(742,600),(760,565),(790,535),(830,522),(880,515),(915,518),(935,535),(940,570),(932,605),(912,628),(880,636),(820,642),(770,646)],
 'leg':[(775,692),(835,690),(870,768),(838,778),(810,800),(795,760),(785,725)],
 'boot':[(700,880),(718,845),(745,815),(780,790),(820,776),(860,772),(880,790),(900,830),(922,870),(918,905),(895,930),(865,955),(810,970),(760,978),(712,975),(696,955),(692,915)],
}
def smask(poly):
    im=Image.new('L',(W,H),0); ImageDraw.Draw(im).polygon(poly,fill=1); return np.array(im,bool)
S={k:smask(v) for k,v in SIDE.items()}
# тело сбоку = розовый контур пользователя (узкое, наклонено вперёд);
# между розовой диагональю и передней чёрной линией — рука
BODY=[(750,322),(800,312),(885,308),(872,340),(850,395),(825,450),(805,500),(830,545),(860,600),(890,660),(880,668),
      (840,668),(760,670),(730,660),(705,635),(680,580),(660,525),(655,500),(675,480),(705,465),(725,430),(740,380),(745,340)]
body=smask(BODY); rr=np.arange(H)[:,None]
S['shirt']=body&(rr<=500); S['belt']=body&(rr>=470)&(rr<=515); S['shorts']=body&(rr>=500)
S['undershirt']=S['shirt']
S['arm']=smask([(835,335),(885,330),(900,400),(912,480),(905,510),(880,520),(840,512),(780,505),(800,450),(820,395)])
DS=1; X0,X1=60,614; Y0,Y1=630,950; Z0,Z1=40,1000
xs=np.arange(X0,X1,DS); ys=np.arange(Y0,Y1,DS); zs=np.arange(Z0,Z1,DS)
lab=np.zeros((len(xs),len(ys),len(zs)),np.int16)
XX,YY=np.meshgrid(xs,ys,indexing='ij'); NEXP=2.4
order=['leg_L','leg_R','arm_L','arm_R','neck','ear_L','ear_R','shorts','belt','shirt','boot_L','boot_R','glove_L','glove_R','head']
from scipy.ndimage import gaussian_filter1d, median_filter
def fill(name,fm,sm,li):
    fr=np.nonzero(fm.any(1))[0]; sr=np.nonzero(sm.any(1))[0]
    f0,f1,s0,s1=fr.min(),fr.max(),sr.min(),sr.max()
    ks,P=[],[]
    for k,z in enumerate(zs):
        if z<f0 or z>f1: continue
        row=np.nonzero(fm[z])[0]
        s=int(round(s0+(z-f0)*(s1-s0)/max(f1-f0,1)))   # высоты вида сбоку подгоняются под вид спереди
        col=np.nonzero(sm[s])[0]
        if len(row)==0 or len(col)==0: continue
        ks.append(k); P.append((row.min()-0.5,row.max()+0.5,col.min()-0.5,col.max()+0.5))
    P=np.array(P,float)
    if len(P)>7:   # убрать дрожание руки рисунка (~0,2 мм), форма контура остаётся
        P=median_filter(P,size=(9,1),mode='nearest'); P=gaussian_filter1d(P,3.2,axis=0,mode='nearest')
    for k,(a,b,c,d) in zip(ks,P):
        e=(np.abs(XX-(a+b)/2)/((b-a)/2))**NEXP+(np.abs(YY-(c+d)/2)/((d-c)/2))**NEXP<=1
        lab[:,:,k][e]=li
for i,n in enumerate(order,1):
    base=n.split('_')[0]
    fm=F[n] if n!='shirt' else (F['shirt']|F['vee'])
    fill(n,fm,S[base],i)
# футболка в V-вырезе: передняя половина толщины торса там, где спереди виден вырез
UI=len(order)+1; order.append('undershirt')
vee=F['vee'][np.ix_(zs,xs)].T          # (x,z)
sh=lab==order.index('shirt')+1
for xi,zi in zip(*np.nonzero(vee)):
    col=np.nonzero(sh[xi,:,zi])[0]
    if len(col): lab[xi,col[col>=(col.min()+col.max())/2],zi]=UI
# перемычки в стыках, где детали только касаются (подмышка, под головой)
def bridge(a,b,r):
    A=lab==order.index(a)+1; B=lab==order.index(b)+1
    if not A.any() or not B.any(): return
    idx=np.argwhere(A|B); lo=np.maximum(idx.min(0)-r,0); hi=idx.max(0)+r+1
    sl=tuple(slice(l,h) for l,h in zip(lo,hi))
    da=nd.distance_transform_edt(~A[sl]); db=nd.distance_transform_edt(~B[sl])
    add=(da<=r)&(db<=r)&(lab[sl]==0)
    cc,_=nd.label(add|A[sl]); keep=np.unique(cc[A[sl]]); add&=np.isin(cc,keep[keep>0])
    sub=lab[sl]; sub[add]=order.index(a)+1; print('bridge',a,b,int(add.sum()))
for a,b in (('arm_L','shirt'),('arm_R','shirt'),('neck','shirt'),('neck','head'),('leg_L','boot_L'),('leg_R','boot_R'),('leg_L','shorts'),('leg_R','shorts')):
    bridge(a,b,14 if a.startswith('arm') else 8)
np.savez_compressed(os.path.join(OUT,'labels.npz'),lab=lab,order=np.array(order))
# --- меши (мм): высота 100 мм = 901 px скетча
MM=100/901; V=DS*MM
def to_mm(v):  # индексы -> мм: X вправо, Y назад (-глубина), Z вверх
    x=(xs[0]+v[:,0]*DS-340)*MM; y=-(ys[0]+v[:,1]*DS-810)*MM; z=(983-(zs[0]+v[:,2]*DS))*MM
    return np.c_[x,y,z]
def mesh(m):
    idx=np.argwhere(m); lo=np.maximum(idx.min(0)-4,0); hi=idx.max(0)+5
    sub=np.pad(m[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]],3)
    d=nd.distance_transform_edt(~sub)-nd.distance_transform_edt(sub)
    d=nd.gaussian_filter(d.astype(np.float32),2.4)
    v,f,_,_=marching_cubes(d,0.0); v=v+lo-3
    t=trimesh.Trimesh(to_mm(v),f[:,::-1],process=True); trimesh.smoothing.filter_taubin(t,iterations=12)
    p=t.split(only_watertight=False); t=max(p,key=lambda q:len(q.faces))
    if t.volume<0: t.invert()
    return t
for i,n in enumerate(order,1):
    m=lab==i
    c,k=nd.label(m)
    if k>1: m=c==(np.argmax(np.bincount(c.ravel())[1:])+1)
    t=mesh(m); t.export(os.path.join(OUT,n+'.stl')); print(n,len(t.faces),t.is_watertight)
w=mesh(lab>0)
try:
    import fast_simplification as fs
    if len(w.faces)>1500000:
        v,f=fs.simplify(w.vertices,w.faces,target_reduction=1-1500000/len(w.faces)); w=trimesh.Trimesh(v,f)
except ImportError: print('no decimation')
w.export(os.path.join(OUT,'whole.stl')); print('whole',w.bounds.round(1).tolist(),w.is_watertight)
