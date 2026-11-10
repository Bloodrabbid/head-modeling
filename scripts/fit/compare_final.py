import os
"""Reference vs the shipped head (LOD0 with its real materials), rendered with the cameras fitted to each screenshot.
usage: python3 compare_final.py head_source.blend cams.json out.jpg [scale]"""
import sys, os, json
import numpy as np, bpy
from mathutils import Matrix
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import render_util as R

src, cjs, out = sys.argv[1:4]; SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
REFS = {'front': (P.REF['front'], 0.5), 'right': (P.REF['right'], 1.0),
        'left': (P.REF['left'], 1.0)}
S1 = json.load(open(P.WORK + 'stage1.json')); O = np.array(S1['origin_mh'])
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
CAMS = {c['name']: c for c in json.load(open(cjs))}
bpy.ops.wm.open_mainfile(filepath=src)
sc = bpy.context.scene
for o in bpy.data.objects:
    if o.type == 'MESH': o.hide_render = not o.name.endswith('_LOD0')
    if o.type == 'CAMERA' or o.type == 'LIGHT': bpy.data.objects.remove(o)
R.studio(sc, world=0.45)
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = int(os.environ.get('SAMPLES', '48')); sc.cycles.use_denoising = True
sc.render.film_transparent = False
rows = []
for name, (path, s) in REFS.items():
    cam = CAMS[name]
    Rm = np.array(cam['R']); t = np.array(cam['t']); f = cam['f']; w, h = cam['w'], cam['h']
    Rb = Rm @ A.T; tb = (Rm @ O + t) / 10.0; C = -Rb.T @ tb * json.load(open(P.CONFIG)).get('export_scale', 1.0)
    cd = bpy.data.cameras.new('c'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0; cd.lens = f * 36.0 / w; cd.clip_start = 0.01
    co = bpy.data.objects.new('c', cd); sc.collection.objects.link(co); sc.camera = co
    M4 = Matrix.Identity(4); RotB = Rb.T @ np.diag([1, -1, -1])
    for i in range(3):
        for j in range(3): M4[i][j] = RotB[i, j]
        M4[i][3] = C[i]
    co.matrix_world = M4
    sc.render.resolution_x, sc.render.resolution_y = int(w * SCALE), int(h * SCALE)
    p = P.FIT + f'_final_{name}.png'; sc.render.filepath = p
    bpy.ops.render.render(write_still=True); bpy.data.objects.remove(co)
    ours = Image.open(p).convert('RGB')
    ref = Image.open(path).convert('RGB'); ref = ref.resize(ours.size)
    H = 360; rows.append([im.resize((int(im.width * H / im.height), H)) for im in (ref, ours)])
W = sum(i.width for i in rows[0]); S = Image.new('RGB', (W, 360 * len(rows)))
for k, row in enumerate(rows):
    x = 0
    for im in row: S.paste(im, (x, 360 * k)); x += im.width
S.save(out, quality=88)
print('saved', out)
