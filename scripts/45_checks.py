import os
"""Functional checks on the shipped LOD0 (from head_source.blend):
 - teeth visible from 5 cameras for key ARKit combinations (closed-mouth poses must show none),
 - eye pivots: eyeball sphere centre vs eye bone heads vs the rotation centre of the eyeLook keys,
 - a contact sheet of all 52 blend shapes at full value."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import render_util as R, view as Vw
OUT = P.OUT + ''; W = P.WORK + ''
bpy.ops.wm.open_mainfile(filepath=OUT + 'head_source.blend')
sc = bpy.context.scene
L0 = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.endswith('_LOD0')]
O = {o.name.replace('_LOD0', ''): o for o in L0}
ARM = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
def pose(**kw):
    for o in L0:
        if o.data.shape_keys:
            for kb in o.data.shape_keys.key_blocks[1:]: kb.value = kw.get(kb.name, 0.0)
def world_verts(o):
    dg = bpy.context.evaluated_depsgraph_get(); e = o.evaluated_get(dg); me = e.to_mesh()
    V = np.array([(e.matrix_world @ v.co)[:] for v in me.vertices]); F = [list(p.vertices) for p in me.polygons]; e.to_mesh_clear()
    return V, F
rep = {}
# ---- teeth visibility
H = O['Head']; Vh, _ = world_verts(H); top = Vh[:, 2].max(); ctr = np.array([0.0, Vh[:, 1].min() + 0.05, top - 0.17])
cams = [ctr + 0.6 * np.array([math.sin(math.radians(a)), -math.cos(math.radians(a)) * math.cos(math.radians(e)), math.sin(math.radians(e))])
        for a, e in ((0, 0), (30, 0), (-30, 0), (40, -10), (-40, -10), (60, 5), (0, -15), (0, -30))]
COMBOS = {'neutral': {}, 'jawOpen': {'jawOpen': 1}, 'jawOpen+mouthClose': {'jawOpen': 1, 'mouthClose': 1}, 'jawOpen.5+mouthClose.5': {'jawOpen': .5, 'mouthClose': .5},
          'jawLeft': {'jawLeft': 1}, 'jawRight': {'jawRight': 1}, 'jawForward': {'jawForward': 1},
          'jawOpen.5+jawLeft': {'jawOpen': 0.5, 'jawLeft': 1}, 'smile': {'mouthSmileLeft': 1, 'mouthSmileRight': 1},
          'pucker': {'mouthPucker': 1}, 'funnel': {'mouthFunnel': 1}, 'rollLower+Upper': {'mouthRollLower': 1, 'mouthRollUpper': 1},
          'press': {'mouthPressLeft': 1, 'mouthPressRight': 1}, 'frown': {'mouthFrownLeft': 1, 'mouthFrownRight': 1},
          'cheekPuff': {'cheekPuff': 1}, 'mouthLeft': {'mouthLeft': 1}, 'shrugLower': {'mouthShrugLower': 1}}
CLOSED = {'neutral', 'jawOpen+mouthClose', 'jawOpen.5+mouthClose.5', 'jawLeft', 'jawRight', 'jawForward', 'smile', 'pucker', 'rollLower+Upper', 'press', 'frown', 'cheekPuff', 'mouthLeft', 'shrugLower'}
vis = {}
for nm, kw in COMBOS.items():
    pose(**kw); bpy.context.view_layer.update()
    allV, allF, owner = [], [], []
    for o in L0:
        if o.name.startswith(('Hair', 'Eyelashes', 'Eyebrows')): continue
        V, F = world_verts(o); off = sum(len(v) for v in allV)
        allV.append(V); allF += [[i + off for i in f] for f in F]; owner += [o.name] * len(F)
    AV = np.concatenate(allV); bvh = BVHTree.FromPolygons([Vector(v) for v in AV], allF)
    Vt, _ = world_verts(O['Teeth']); seen = 0
    for v in Vt[::3]:
        for c in cams:
            d = v - c; dist = np.linalg.norm(d); d /= dist
            hit = bvh.ray_cast(Vector(c), Vector(d), dist + 1e-3)
            if hit[0] is not None and owner[hit[2]] == 'Teeth_LOD0' and np.linalg.norm(np.array(hit[0]) - v) < 0.0015: seen += 1; break
    vis[nm] = seen
pose()
rep['teeth_vertices_visible'] = vis
rep['teeth_check_closed_mouth_ok'] = all(vis[k] == 0 for k in CLOSED)
print('teeth visible:', vis)
# ---- eye pivots
def sphere_fit(P):
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Q = P[keep]; A_ = np.c_[2 * Q, np.ones(len(Q))]; sol = np.linalg.lstsq(A_, (Q * Q).sum(1), rcond=None)[0]
        c = sol[:3]; r = np.sqrt(sol[3] + c @ c); res = np.abs(np.linalg.norm(P - c, axis=1) - r); keep = res <= np.quantile(res, 0.8)
    return c, r
E = O['Eyes']; B0 = np.array([v.co[:] for v in E.data.shape_keys.key_blocks[0].data])
eyes = {}
for sgn, sd, bn in ((1, 'Left', 'eye_L'), (-1, 'Right', 'eye_R')):
    m = np.sign(B0[:, 0]) == sgn; c, r = sphere_fit(B0[m])
    K = np.array([v.co[:] for v in E.data.shape_keys.key_blocks['eyeLookOut' + sd].data])[m]
    # a rotation about the sphere centre keeps every vertex at the same distance from it
    drift = float(np.abs(np.linalg.norm(K - c, axis=1) - np.linalg.norm(B0[m] - c, axis=1)).max())
    bh = np.array((ARM.matrix_world @ ARM.data.bones[bn].head_local)[:])
    ax = np.array([0, 0, 1.0])
    eyes[sd] = {'sphere_centre': c.round(4).tolist(), 'diameter_mm': round(float(2 * r * 1000), 1),
                'bone_head_to_centre_mm': round(float(np.linalg.norm(bh - c) * 1000), 2),
                'eyeLookOut_distance_drift_mm': round(drift * 1000, 2)}
rep['eyes'] = eyes; print('eyes:', eyes)
# ---- symmetry of the head: neutral and Left/Right pairs
from scipy.spatial import cKDTree
mi = cKDTree(B0h := np.array([v.co[:] for v in H.data.shape_keys.key_blocks[0].data]) * np.array([-1, 1, 1])).query(np.array([v.co[:] for v in H.data.shape_keys.key_blocks[0].data]))[1]
Bh = np.array([v.co[:] for v in H.data.shape_keys.key_blocks[0].data]); kbh = H.data.shape_keys.key_blocks
asym = {'neutral_max_mm': round(float(np.linalg.norm(Bh - Bh[mi] * np.array([-1, 1, 1]), axis=1).max() * 1000), 2)}
worst = 0
for k in kbh:
    if k.name.endswith('Left') and k.name.replace('Left', 'Right') in kbh:
        L_ = np.array([v.co[:] for v in k.data]) - Bh; R_ = np.array([v.co[:] for v in kbh[k.name.replace('Left', 'Right')].data]) - Bh
        worst = max(worst, float(np.linalg.norm(R_ - L_[mi] * np.array([-1, 1, 1]), axis=1).max()))
asym['left_right_pair_max_mm'] = round(worst * 1000, 2); rep['symmetry'] = asym; print('symmetry:', asym)
# ---- lashes/brows vs skin in combinations
def clearance(proxy, combos):
    out = {}
    for nm, kw in combos.items():
        pose(**kw); bpy.context.view_layer.update()
        Vh2, Fh2 = world_verts(H); bv = BVHTree.FromPolygons([Vector(v) for v in Vh2], Fh2); Vp, _ = world_verts(O[proxy]); m = 1.0
        for p in Vp:
            loc, nrm, _, dist = bv.find_nearest(Vector(p))
            if loc is not None and dist < 0.01: m = min(m, (Vector(p) - loc).dot(nrm))
        out[nm] = round(m * 1000, 2)
    pose(); return out
rep['lash_min_clearance_mm'] = clearance('Eyelashes', {'neutral': {}, 'blinkL': {'eyeBlinkLeft': 1}, 'blink+squint': {'eyeBlinkLeft': 1, 'eyeBlinkRight': 1, 'eyeSquintLeft': 1, 'eyeSquintRight': 1}, 'blink+lookDown': {'eyeBlinkLeft': 1, 'eyeLookDownLeft': 1}})
rep['brow_min_clearance_mm'] = clearance('Eyebrows', {'neutral': {}, 'browDown': {'browDownLeft': 1, 'browDownRight': 1}, 'browUp': {'browInnerUp': 1, 'browOuterUpLeft': 1, 'browOuterUpRight': 1}})
print('lashes:', rep['lash_min_clearance_mm'], 'brows:', rep['brow_min_clearance_mm'])
hb = ARM.data.bones['head'].head_local; rep['head_pivot_m'] = [round(x, 4) for x in hb]; print('head pivot', rep['head_pivot_m'])
# ---- 52 shapes sheet
R.studio(sc, world=0.4)
for o in bpy.data.objects:
    if o.type == 'MESH': o.hide_render = not o.name.endswith('_LOD0')
names = sorted({kb.name for o in L0 if o.data.shape_keys for kb in o.data.shape_keys.key_blocks[1:]})
tiles = []
from PIL import Image, ImageDraw
for nm in names:
    pose(**{nm: 1.0, **({'jawOpen': 0.6} if nm == 'tongueOut' else {})})
    p = Vw.shots(sc, W + 'sk_' + nm, [('q', 20, 0)], (0, Vh[:, 1].min() + 0.06, top - 0.15), dist=0.55, res=(220, 250), samples=12)[0]
    im = Image.open(p).convert('RGB'); ImageDraw.Draw(im).text((4, 4), nm, fill=(255, 255, 0)); tiles.append(im)
pose()
cols = 9; rows = (len(tiles) + cols - 1) // cols
S = Image.new('RGB', (220 * cols, 250 * rows), (40, 40, 40))
for k, t in enumerate(tiles): S.paste(t, ((k % cols) * 220, (k // cols) * 250))
S.save(OUT + 'preview_all_shapes.png'); rep['shape_sheet'] = len(names)
json.dump(rep, open(OUT + 'checks.json', 'w'), indent=1)
print('checks done, shapes on sheet:', len(names))
