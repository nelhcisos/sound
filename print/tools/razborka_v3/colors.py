"""Цвета для выбора швов: шов «кита» идёт по стыку двух материалов/цветов, где есть ступенька формы.

1) Модель с текстурой (GLB):
   python3 colors.py out.png --glb src.glb [--up y] [--rotz -90]
   Тот же поворот/масштаб, что у prep.py (мастер 100 мм, лицо -> -Y). 4 вида в цветах текстуры,
   чёрным — линии смены цвета, сетка в мм: по ней видно, на какой высоте кончается рукав, подол, волосы.

2) Текстуры нет — референс рядом с моделью на общей сетке высот:
   python3 colors.py out.png --master master.stl --ref ref.jpg --crop x0,y0,x1,y1
   crop — рамка фигурки на картинке в пикселях: от макушки (y0) до низа подставки (y1).
   Красные линии уровней проходят через вид спереди и через референс — видно, где на модели смена цвета.
"""
import sys, os, argparse
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3lib import render, sheet
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--glb"); ap.add_argument("--up", default="y"); ap.add_argument("--rotz", type=float, default=0)
ap.add_argument("--master"); ap.add_argument("--ref"); ap.add_argument("--crop")
ap.add_argument("--grid", type=float, default=5); ap.add_argument("--W", type=int, default=520)
ap.add_argument("--thr", type=float, default=0.18, help="порог смены цвета (0..1)")
a = ap.parse_args()

if a.glb:
    sc = trimesh.load(a.glb, force="scene")
    V, F, C = [], [], []; off = 0
    for name, g in sc.geometry.items():
        g = g.copy()
        tr = sc.graph.get(name)[0] if name in sc.graph.nodes else np.eye(4)
        try:
            vis = g.visual.to_color() if hasattr(g.visual, "to_color") else g.visual  # текстура -> цвета вершин
            vc = np.asarray(vis.vertex_colors)[:, :3] / 255.0
        except Exception as ex:
            print("цвета не прочитаны (%s) — серый" % ex); vc = np.full((len(g.vertices), 3), 0.8)
        g.apply_transform(tr)
        V.append(g.vertices); F.append(g.faces + off); C.append(vc); off += len(g.vertices)
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False); vcol = np.vstack(C)
    if a.up == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if a.rotz:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(a.rotz), [0, 0, 1]))
    m.apply_scale(100.0 / m.extents[2])
    m.apply_translation([-(m.bounds[0][0] + m.bounds[1][0]) / 2, -(m.bounds[0][1] + m.bounds[1][1]) / 2, -m.bounds[0][2]])
    fcol = vcol[m.faces].mean(1)
    # линии смены цвета: грани, у соседа которых цвет заметно другой
    mm = m.copy(); mm.merge_vertices(merge_tex=True, merge_norm=True)  # соседство граней через склеенные вершины
    adj = mm.face_adjacency
    diff = np.linalg.norm(fcol[adj[:, 0]] - fcol[adj[:, 1]], axis=1)
    edge = np.zeros(len(m.faces), bool); e = adj[diff > a.thr]; edge[e[:, 0]] = True; edge[e[:, 1]] = True
    fc = fcol.copy(); fc[edge] = 0.0
    ims = render([("m", m, fc * 255)], W=a.W, grid=a.grid)
    sheet(ims, 4, W=a.W).save(a.out); print("ok", a.out, "граней на границах цвета:", int(edge.sum()))

elif a.master and a.ref:
    m = trimesh.load(a.master, force="mesh")
    lo, hi = m.bounds; C = (lo + hi) / 2
    ext = max(np.ptp(m.vertices[:, 0]), np.ptp(m.vertices[:, 1]), np.ptp(m.vertices[:, 2]))
    S = (a.W - 30) / ext
    im = render([("m", m, (200, 200, 210))], views=("front",), W=a.W, grid=a.grid, scale=S, center=C)[0]
    ytop = int(a.W / 2 - (hi[2] - C[2]) * S); ybot = int(a.W / 2 - (lo[2] - C[2]) * S)
    ref = Image.open(a.ref).convert("RGB")
    if a.crop:
        x0, y0, x1, y1 = [int(float(t)) for t in a.crop.split(",")]; ref = ref.crop((x0, y0, x1, y1))
    h = ybot - ytop; ref = ref.resize((max(1, int(ref.width * h / ref.height)), h))
    out = Image.new("RGB", (a.W + ref.width + 20, a.W), "white"); out.paste(im, (0, 0)); out.paste(ref, (a.W + 10, ytop))
    d = ImageDraw.Draw(out)
    for z in np.arange(0, hi[2] + a.grid, a.grid):
        yy = int(a.W / 2 - (z - C[2]) * S)
        if 0 <= yy < a.W:
            for x in range(a.W - 40, out.width, 6): d.line([(x, yy), (x + 3, yy)], fill=(220, 0, 0))
            d.text((out.width - 28, yy - 10), "%g" % z, fill=(220, 0, 0))
    out.save(a.out); print("ok", a.out)
else:
    raise SystemExit("нужно --glb или --master + --ref")
