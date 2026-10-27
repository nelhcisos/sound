"""Картинка «покраска деталей»: 4 вида + легенда. usage: labels_views.py model.glb L.npy out.png"""
import os, numpy as np, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_front, splat
from PIL import Image, ImageDraw, ImageFont
m, fc = load_front(sys.argv[1]); L = np.load(sys.argv[2])
names = ["Платформа","Ботинок Л","Ботинок П","Штанина Л","Штанина П","Тело","Рукав Л","Рукав П","Ладони","Шея","Голова","Волосы"]
pal = np.array([[1,1,1],[0.85,0.85,0.9],[0.15,0.35,0.9],[0.1,0.75,0.95],[0.55,0.2,0.85],[0.95,0.4,0.8],
 [0.3,0.3,0.3],[0.2,0.75,0.3],[0.65,0.9,0.2],[1.0,0.85,0.55],[0.95,0.6,0.1],[1.0,0.92,0.75],[0.9,0.1,0.1]])
ims=[splat(m, pal[L], v, 0.15)[0] for v in ["front","left","back","right"]]
h=max(i.shape[0] for i in ims)
row=np.concatenate([np.pad(i,((0,h-i.shape[0]),(8,8),(0,0)),constant_values=1) for i in ims],1)
img=Image.fromarray((row*255).astype(np.uint8)); W=img.size[0]
F=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",22)
out=Image.new("RGB",(W,h+40+110),"white"); out.paste(img,(0,40)); d=ImageDraw.Draw(out)
for i,t in enumerate(["спереди","слева","сзади","справа"]): d.text((i*W//4+W//8,8),t,fill=(0,0,0),font=F,anchor="ma")
for i,n in enumerate(names):
    x=20+(i%6)*(W//6); y=h+55+(i//6)*48
    d.rectangle([x,y,x+30,y+30],fill=tuple((pal[i+1]*255).astype(int)),outline=(0,0,0))
    d.text((x+40,y+15),f"{i+1}. {n}",fill=(0,0,0),font=F,anchor="lm")
out.save(sys.argv[3])
