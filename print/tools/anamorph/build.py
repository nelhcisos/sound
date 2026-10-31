"""Сборка сцены: камни → проекция узора → площадки мха → рендеры.

python3 build.py OUTDIR [render]
"""
import sys, json, os, subprocess, time
import numpy as np
import trimesh
from anamorph import (front_facets, moss_regions, moss_solid, convex_manifold, from_manifold, unit)
import scene_moss_line as SC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)

ROCK = "#d8c7a1"
CLIFF = "#d3c19a"
MOSS = "#5b9a2e"


def mesh_dict(V, F, color, **kw):
    d = dict(V=np.round(np.asarray(V, float), 4).ravel().tolist(),
             F=np.asarray(F, int).ravel().tolist(), color=color)
    d.update(kw)
    return d


def compute(seed=SC.SEED, margin=0.8, verbose=True):
    import manifold3d as mf
    from shapely.ops import unary_union
    t0 = time.time()
    stones = SC.build_stones(seed)
    cam = SC.camera()
    pat, curve, w = SC.pattern(cam)
    vis_stones = [s.clip_z(0.0) for s in stones]
    regions, visible, sil, front = moss_regions(cam, vis_stones, pat, margin)
    cover = pat.difference(unary_union([v for v in visible if v is not None]))
    hulls = [convex_manifold(s) for s in vis_stones]
    pads = {}
    for k, M in enumerate(regions):
        if M is None:
            continue
        s = vis_stones[k]
        solid = moss_solid(cam, s, M, front_facets(cam, s))
        if solid is None:
            continue
        near = [convex_manifold(vis_stones[j].grow(0.05)) for j in range(len(stones))
                if j != k and sil[j].distance(sil[k]) < 0.01]
        if near:
            solid = solid - mf.Manifold.batch_boolean(near, mf.OpType.Add)
        if solid.volume() > 1e-4:
            pads[k] = solid
    # профиль глубины вдоль осевой линии
    prof = []
    for p in curve[::6]:
        D = cam.rays(p)[0]
        best = (np.inf, -1)
        for j, s in enumerate(vis_stones):
            t = s.ray_enter(cam.E, D)
            if t is not None and t > 0 and t < best[0]:
                best = (t, j)
        X = cam.E + best[0] * D
        prof.append((best[1], X[1]))
    ids = [j for j, _ in prof]
    jumps = sum(1 for a, b in zip(prof, prof[1:]) if abs(a[1] - b[1]) > 6)
    if verbose:
        print(f"stones={len(stones)} moss on {len(pads)}; uncovered={cover.area:.2e} "
              f"(pattern {pat.area:.2e}); depth y {min(y for _, y in prof):.0f}..{max(y for _, y in prof):.0f}, "
              f"jumps>6mm: {jumps}, distinct stones on line: {len(set(ids))}; {time.time()-t0:.1f}s")
    return dict(stones=stones, vis=vis_stones, cam=cam, pat=pat, curve=curve, pads=pads,
                regions=regions, prof=prof, jumps=jumps)


def magic_cam(cam, stones, w=1800):
    V = np.vstack([s.clip_z(0).geometry()[0] for s in stones])
    uv, _ = cam.project(V)
    hu, hv = np.abs(uv[:, 0]).max() * 1.04, np.abs(uv[:, 1]).max() * 1.12
    h = int(round(w * hv / hu))
    return dict(name="magic", E=cam.E.tolist(), T=SC.LOOK.tolist(), vfov=float(np.degrees(2 * np.arctan(hv))), w=w, h=h)


def scene_json(stones, pads, style="game"):
    meshes = []
    for k, s in enumerate(stones):
        V, F = s.clip_z(-0.5).mesh()
        meshes.append(mesh_dict(V, F, CLIFF if s.meta.get("kind") == "cliff" else ROCK, reflect=True))
    for k, solid in pads.items():
        Vp, Fp = from_manifold(solid)
        meshes.append(mesh_dict(Vp, Fp, MOSS, reflect=True, rough=1.0))
    S = dict(meshes=meshes)
    if style == "game":
        S.update(bg="#62a6d9", water=dict(center=[0, 40], size=[600, 600], color="#3d86ad", opacity=0.62))
    else:
        S.update(bg="#f2f0ea")
    return S


def render(S, cams, outdir):
    sf, cf = os.path.join(outdir, "_scene.json"), os.path.join(outdir, "_cams.json")
    json.dump(S, open(sf, "w"))
    json.dump(cams, open(cf, "w"))
    env = dict(os.environ, NODE_PATH=subprocess.check_output(["npm", "root", "-g"]).decode().strip(),
               THREE=os.environ.get("THREE", "/tmp/claude-0/-home-user/f561068c-8a2f-5035-9042-8362a5d9ecce/scratchpad/r/node_modules/three/build/three.module.js"))
    subprocess.run(["node", os.path.join(HERE, "render.js"), sf, cf, outdir], check=True, env=env)
    os.remove(sf)
    os.remove(cf)


if __name__ == "__main__":
    R = compute()
    stones, cam, pads = R["stones"], R["cam"], R["pads"]
    if "render" in sys.argv:
        mc = magic_cam(cam, stones)
        E, T = np.array(mc["E"]), np.array(mc["T"])
        cams = [mc,
                dict(mc, name="right", E=(E + [95, 30, 10]).tolist()),
                dict(mc, name="left", E=(E + [-95, 30, 25]).tolist()),
                dict(mc, name="high", E=[-60, -110, 170], T=[0, 40, 10], vfov=44, w=1600, h=1000)]
        render(scene_json(stones, pads), cams, OUT)
