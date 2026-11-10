import os
"""Render several morph-weight variants of the head with one fitted camera and crop a region, for quick A/B looks.
usage: python3 variants.py weights.json cams.json view variants.json crop(x0,y0,x1,y1 as fractions) out.jpg
variants.json: [{"name": "...", "set": {"target": weight, ...}}, ...]  (weight 0 removes a target)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy
from mathutils import Matrix
from PIL import Image, ImageDraw
import mhlib as M, render_util as R, view as Vw
from scipy.spatial import cKDTree

wjs, cjs, view, vjs, crop, out = sys.argv[1:7]
crop = [float(x) for x in crop.split(',')]
CFG = json.load(open(wjs)); CAM = {c['name']: c for c in json.load(open(cjs))}[view]; VAR = json.load(open(vjs))
S1 = json.load(open(P.WORK + 'stage1.json')); O = np.array(S1['origin_mh'])
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
def conv(P): return (np.asarray(P) - O) @ A.T * 0.1
V0 = M.load_obj(M.DATA + '/3dobjs/base.obj')[0]
def shape(tw):
    TW = {}
    for k, w in tw.items():
        if not w: continue
        if '/lr-' in k: TW[k.replace('/lr-', '/l-')] = w; TW[k.replace('/lr-', '/r-')] = w
        else: TW[k] = w
    return M.apply_targets(V0, TW)

bpy.ops.wm.open_mainfile(filepath=P.WORK + 'stage1.blend')
sc = bpy.context.scene
head = bpy.data.objects['Head']
if head.data.shape_keys: head.shape_key_clear()
Vn_saved = np.load(P.WORK + 'Vn.npy'); cand = np.array(list(head['orig_index']))
HV0 = np.array([v.co[:] for v in head.data.vertices]); used = cand[cKDTree(conv(Vn_saved[cand])).query(HV0)[1]]
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o.name not in ('Head', 'Eyes', 'Eyebrows', 'Eyelashes'): bpy.data.objects.remove(o)
    elif o.type == 'MESH' and o.data.shape_keys: o.shape_key_clear()
Vw.simple_materials(); R.studio(sc, world=0.45)
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 24; sc.cycles.use_denoising = True
Rm = np.array(CAM['R']); t = np.array(CAM['t']); f = CAM['f']; w, h = CAM['w'], CAM['h']
Rb = Rm @ A.T; tb = (Rm @ O + t) / 10.0; C = -Rb.T @ tb
cd = bpy.data.cameras.new('c'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0; cd.lens = f * 36.0 / w; cd.clip_start = 0.01
co = bpy.data.objects.new('c', cd); sc.collection.objects.link(co); sc.camera = co
M4 = Matrix.Identity(4); RotB = Rb.T @ np.diag([1, -1, -1])
for i in range(3):
    for j in range(3): M4[i][j] = RotB[i, j]
    M4[i][3] = C[i]
co.matrix_world = M4
sc.render.resolution_x, sc.render.resolution_y = w, h
sc.render.use_border = True; sc.render.use_crop_to_border = True
sc.render.border_min_x, sc.render.border_max_x = crop[0], crop[2]
sc.render.border_min_y, sc.render.border_max_y = 1 - crop[3], 1 - crop[1]
tiles = []
REFP = {'front': (P.REF['front'], 0.5), 'right': (P.REF['right'], 1.0), 'left': (P.REF['left'], 1.0)}[view]
ref = Image.open(REFP[0]).convert('RGB').resize((w, h)).crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
ImageDraw.Draw(ref).text((4, 2), 'REFERENCE', fill=(255, 255, 0)); tiles.append(ref)
for var in VAR:
    tw = dict(CFG['targets']); tw.update(var['set'])
    Vf = shape(tw)
    head.data.vertices.foreach_set('co', conv(Vf[used]).astype(np.float32).ravel()); head.data.update()
    for name in ('Eyes', 'Eyebrows', 'Eyelashes'):
        ob = bpy.data.objects[name]
        ob.data.vertices.foreach_set('co', conv(M.fit_proxy(M.load_mhclo(M.PACK + '/' + ob['mhclo'] + '.mhclo'), Vf)).astype(np.float32).ravel()); ob.data.update()
    p = P.FIT + f'_var_{var["name"]}.png'; sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    im = Image.open(p).convert('RGB').resize(ref.size); ImageDraw.Draw(im).text((4, 2), var['name'], fill=(255, 255, 0)); tiles.append(im)
cols = min(4, len(tiles)); rows = (len(tiles) + cols - 1) // cols
S = Image.new('RGB', (tiles[0].width * cols, tiles[0].height * rows))
for k, tl in enumerate(tiles): S.paste(tl, ((k % cols) * tl.width, (k // cols) * tl.height))
if S.width < 1200: S = S.resize((S.width * 2, S.height * 2))
S.save(out, quality=88); print('saved', out)
