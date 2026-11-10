import os
"""Contact sheet of proxy options (hair or brows) on the fitted head, rendered with the fitted reference cameras.
usage: python3 preview_assets.py weights.json cams.json kind name1,name2,... out.jpg [views]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy
from mathutils import Matrix
from PIL import Image
import mhlib as M, render_util as R, view as Vw

wjs, cjs, kind, names, out = sys.argv[1:6]
VIEWS = sys.argv[6].split(',') if len(sys.argv) > 6 else ['front', 'right']
CFG = json.load(open(wjs)); CAMS = {c['name']: c for c in json.load(open(cjs))}
S1 = json.load(open(P.WORK + 'stage1.json')); O = np.array(S1['origin_mh'])
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
def conv(P): return (np.asarray(P) - O) @ A.T * 0.1
V0 = M.load_obj(M.DATA + '/3dobjs/base.obj')[0]
TW = {}
for k, w in CFG['targets'].items():
    if '/lr-' in k: TW[k.replace('/lr-', '/l-')] = w; TW[k.replace('/lr-', '/r-')] = w
    else: TW[k] = w
Vf = M.apply_targets(V0, TW)

bpy.ops.wm.open_mainfile(filepath=P.WORK + 'stage1.blend')
sc = bpy.context.scene
from scipy.spatial import cKDTree
head = bpy.data.objects['Head']
if head.data.shape_keys: head.shape_key_clear()
Vn_saved = np.load(P.WORK + 'Vn.npy'); cand = np.array(list(head['orig_index']))
HV0 = np.array([v.co[:] for v in head.data.vertices]); used = cand[cKDTree(conv(Vn_saved[cand])).query(HV0)[1]]
head.data.vertices.foreach_set('co', conv(Vf[used]).astype(np.float32).ravel())
for name in ('Eyes', 'Eyebrows', 'Eyelashes'):
    ob = bpy.data.objects[name]
    if ob.data.shape_keys: ob.shape_key_clear()
    ob.data.vertices.foreach_set('co', conv(M.fit_proxy(M.load_mhclo(M.PACK + '/' + ob['mhclo'] + '.mhclo'), Vf)).astype(np.float32).ravel())
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o.name in ('Hair', 'Teeth', 'Tongue', 'Eyes_low'): bpy.data.objects.remove(o)
    elif kind == 'brows' and o.name == 'Eyebrows': bpy.data.objects.remove(o)

def add(rel, name):
    clo = M.load_mhclo(M.PACK + '/' + rel + '.mhclo')
    pv, puv, pf, pfuv, _ = M.load_obj(clo['dir'] + '/' + clo['obj_file'])
    P = conv(M.fit_proxy(clo, Vf))
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in P], [], [tuple(f) for f in pf]); me.update()
    ul = me.uv_layers.new(name='UVMap'); ul.data.foreach_set('uv', np.array([puv[t] for fu in pfuv for t in fu], np.float32).ravel())
    for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    mat = M.read_mhmat(os.path.normpath(os.path.join(clo['dir'], clo['material'])))
    ob['texture'] = os.path.normpath(os.path.join(os.path.dirname(os.path.join(clo['dir'], clo['material'])), mat['diffuseTexture'][0]))
    return ob

Vw.simple_materials(); R.studio(sc, world=0.45)
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 24; sc.cycles.use_denoising = True
def set_cam(cam, scale=0.5):
    Rm = np.array(cam['R']); t = np.array(cam['t']); f = cam['f']; w, h = cam['w'], cam['h']
    Rb = Rm @ A.T; tb = (Rm @ O + t) / 10.0; C = -Rb.T @ tb
    cd = bpy.data.cameras.new('c'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0; cd.lens = f * 36.0 / w; cd.clip_start = 0.01
    co = bpy.data.objects.new('c', cd); sc.collection.objects.link(co); sc.camera = co
    M4 = Matrix.Identity(4); RotB = Rb.T @ np.diag([1, -1, -1])
    for i in range(3):
        for j in range(3): M4[i][j] = RotB[i, j]
        M4[i][3] = C[i]
    co.matrix_world = M4
    sc.render.resolution_x, sc.render.resolution_y = int(w * scale), int(h * scale)
    return co

tiles = []
for nm in names.split(','):
    rel = f'{kind if kind != "brows" else "eyebrows"}/{nm}/{nm}'
    ob = add(rel, 'cand'); Vw.simple_materials()
    row = []
    for v in VIEWS:
        co = set_cam(CAMS[v]); p = P.FIT + f'_pv_{nm}_{v}.png'; sc.render.filepath = p
        bpy.ops.render.render(write_still=True); bpy.data.objects.remove(co); row.append(Image.open(p).convert('RGB'))
    bpy.data.objects.remove(ob)
    H = 300; row = [im.resize((int(im.width * H / im.height), H)) for im in row]
    tile = Image.new('RGB', (sum(i.width for i in row), H + 22), (30, 30, 30)); x = 0
    for im in row: tile.paste(im, (x, 22)); x += im.width
    from PIL import ImageDraw; ImageDraw.Draw(tile).text((6, 4), nm, fill=(255, 255, 0))
    tiles.append(tile)
cols = 2; rows_ = (len(tiles) + cols - 1) // cols
S = Image.new('RGB', (tiles[0].width * cols, tiles[0].height * rows_), (0, 0, 0))
for k, t in enumerate(tiles): S.paste(t, ((k % cols) * tiles[0].width, (k // cols) * tiles[0].height))
S.save(out, quality=85)
