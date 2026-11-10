import os
"""Render the current head from given cameras, detect MediaPipe landmarks on the render, and map every landmark
back onto the MakeHuman base mesh (3 vertex ids + barycentric weights) by casting a ray from the camera.

usage: python3 fit_render.py weights.json cams.json out_prefix
  weights.json : {"targets": {...}} in config.json format (lr- keys allowed)
  cams.json    : [{"name", "R" (3x3, OpenCV, MakeHuman space), "t" (3), "f" (px), "w", "h"}]
writes out_prefix.json with, per view, the 2D landmarks found on our render and their surface correspondences."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import mhlib as M, render_util as R, view as Vw

wjs, cjs, outp = sys.argv[1:4]; outp = os.path.abspath(outp)
CFG = json.load(open(wjs)); CAMS = json.load(open(cjs))
S1 = json.load(open(P.WORK + 'stage1.json')); O = np.array(S1['origin_mh'])
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # MakeHuman axes -> Blender axes
def conv(P): return (np.asarray(P) - O) @ A.T * 0.1

V0 = M.load_obj(M.DATA + '/3dobjs/base.obj')[0]
TW = {}
for k, w in CFG['targets'].items():
    if '/lr-' in k: TW[k.replace('/lr-', '/l-')] = w; TW[k.replace('/lr-', '/r-')] = w
    else: TW[k] = w
Vf = M.apply_targets(V0, TW)

bpy.ops.wm.open_mainfile(filepath=P.WORK + 'stage1.blend')
sc = bpy.context.scene
head = bpy.data.objects['Head']
if head.data.shape_keys: head.shape_key_clear()
# the stage-1 head was bisected at the neck, so its vertex order no longer follows orig_index: match by position
from scipy.spatial import cKDTree
Vn_saved = np.load(P.WORK + 'Vn.npy'); cand = np.array(list(head['orig_index']))
HV0 = np.array([v.co[:] for v in head.data.vertices])
dd, nn = cKDTree(conv(Vn_saved[cand])).query(HV0)
used = cand[nn]
print('vertex match: max dist mm', round(float(dd.max() * 1000), 2), 'exact', int((dd < 1e-5).sum()), 'of', len(dd))
head.data.vertices.foreach_set('co', conv(Vf[used]).astype(np.float32).ravel())
for name in ('Eyes', 'Eyebrows', 'Eyelashes'):
    ob = bpy.data.objects[name]
    if ob.data.shape_keys: ob.shape_key_clear()
    P = conv(M.fit_proxy(M.load_mhclo(M.PACK + '/' + ob['mhclo'] + '.mhclo'), Vf))
    ob.data.vertices.foreach_set('co', P.astype(np.float32).ravel())
for o in bpy.data.objects:
    if o.type == 'MESH': o.hide_render = o.name not in ('Head', 'Eyes', 'Eyebrows')
head.data.update()
Vw.simple_materials(); R.studio(sc, world=0.45)
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 16; sc.cycles.use_denoising = True

dg = bpy.context.evaluated_depsgraph_get()
bvh = BVHTree.FromObject(head, dg)
polys = [list(p.vertices) for p in head.data.polygons]
HV = np.array([v.co[:] for v in head.data.vertices])

import mediapipe as mp, cv2
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=P.MP_MODEL),
    num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))

def bary_in_poly(pi, p):
    """Triangle of polygon pi containing point p (fan), with barycentric weights."""
    vs = polys[pi]; best = None
    for k in range(1, len(vs) - 1):
        a, b, c = HV[vs[0]], HV[vs[k]], HV[vs[k + 1]]
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if den == 0: continue
        bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; bu = 1 - bv - bw
        err = -min(bu, bv, bw)
        if best is None or err < best[0]: best = (err, [vs[0], vs[k], vs[k + 1]], [bu, bv, bw])
    return best[1], best[2]

# pupils: the eyeball proxy is bound to the base mesh, so each pupil is a fixed blend of base vertices plus an offset
eclo = M.load_mhclo(M.PACK + '/' + bpy.data.objects['Eyes']['mhclo'] + '.mhclo')
EP = M.fit_proxy(eclo, Vf)
IRIS = {}
for idx, sgn in ((468, -1), (473, 1)):                     # MediaPipe 468 = subject's right iris (MakeHuman -X)
    side = np.where(np.sign(EP[:, 0]) == sgn)[0]
    front = side[np.argsort(-EP[side, 2])[:6]]
    vs = eclo['refs'][front].ravel().tolist(); bw = (eclo['wts'][front] / len(front)).ravel().tolist()
    sc_ = np.ones(3)
    for k, ax in (('x', 0), ('y', 1), ('z', 2)):
        if k in eclo['scale']:
            a_, b_, den = eclo['scale'][k]; sc_[ax] = abs(Vf[a_, ax] - Vf[b_, ax]) / den
    IRIS[idx] = {'v': [int(x) for x in vs], 'b': bw, 'c': (eclo['offs'][front] * sc_).mean(0).tolist(), 'facing': 1.0}

res = {}
for cam in CAMS:
    Rm = np.array(cam['R']); t = np.array(cam['t']); f = cam['f']; w, h = cam['w'], cam['h']
    Rb = Rm @ A.T; tb = (Rm @ O + t) / 10.0                         # same camera, in Blender world (metres)
    C = -Rb.T @ tb
    cd = bpy.data.cameras.new('fitcam'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0
    cd.lens = f * 36.0 / w; cd.clip_start = 0.01
    co = bpy.data.objects.new('fitcam', cd); sc.collection.objects.link(co); sc.camera = co
    M4 = Matrix.Identity(4); RotB = Rb.T @ np.diag([1, -1, -1])
    for i in range(3):
        for j in range(3): M4[i][j] = RotB[i, j]
        M4[i][3] = C[i]
    co.matrix_world = M4
    sc.render.resolution_x, sc.render.resolution_y = w, h; sc.render.resolution_percentage = 100
    png = f"{outp}_{cam['name']}.png"; sc.render.filepath = png
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(co)
    img = cv2.imread(png)
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        print(cam['name'], 'no face on our render'); res[cam['name']] = None; continue
    L2 = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
    corr = []
    for (u, v) in L2[:468]:
        d = Rb.T @ np.array([(u - w / 2) / f, (v - h / 2) / f, 1.0]); d /= np.linalg.norm(d)
        loc, nrm, idx, dist = bvh.ray_cast(Vector(C), Vector(d))
        if loc is None: corr.append(None); continue
        vs, bw = bary_in_poly(idx, np.array(loc))
        facing = float(-(np.array(nrm) @ d))
        corr.append({'v': [int(used[k]) for k in vs], 'b': [float(x) for x in bw], 'facing': round(facing, 3)})
    corr += [None] * (478 - len(corr))
    for idx in IRIS: corr[idx] = IRIS[idx]
    res[cam['name']] = {'pts': L2.round(2).tolist(), 'corr': corr}
    ov = img.copy()
    for x, y in L2: cv2.circle(ov, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(f"{outp}_{cam['name']}_lm.jpg", ov)
    print(cam['name'], 'landmarks', len(L2), 'mapped', sum(c is not None for c in corr))
json.dump(res, open(outp + '.json', 'w'))
os._exit(0)          # MediaPipe's task object crashes in its destructor at interpreter shutdown
