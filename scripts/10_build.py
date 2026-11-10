"""v2, stage 1: character from the MakeHuman base mesh (CC0) shaped toward the reference,
head + neck cut out, proxies fitted (eyes, lashes, brows, teeth, tongue, hair). Saves work/stage1.blend."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bpy
import mhlib as M

import paths as P
W = P.WORK
CFG = json.load(open(P.CONFIG))

V0, UV, F, FUV, G = M.load_obj(M.DATA + '/3dobjs/base.obj')
G = np.array(G)
TW = {}
for k, w in CFG['targets'].items():
    if '/lr-' in k:
        TW[k.replace('/lr-', '/l-')] = w; TW[k.replace('/lr-', '/r-')] = w
    else: TW[k] = w
Vn = M.apply_targets(V0, TW)

# ---- landmarks from the data itself: the vertex a target moves most is the landmark it controls
def peak(tname, axis, side=None, pool=None):
    i, o = M.load_target(M.TGT + '/' + tname + '.target')
    m = np.ones(len(i), bool)
    if side is not None: m &= np.sign(V0[i, 0]) == side
    j = i[m][np.abs(o[m][:, axis]).argmax()]
    return int(j)
LMK = {'mouth_corner': peak('mouth/mouth-angles-up', 1, 1), 'alar': peak('nose/nose-scale-horiz-incr', 0, 1),
       'chin': peak('chin/chin-prominent-incr', 2), 'cheekbone': peak('cheek/l-cheek-bones-incr', 0),
       'nose_tip': peak('nose/nose-scale-depth-incr', 2), 'brow': peak('eyebrows/eyebrows-trans-up', 1, 1)}

# ---- head + neck selection: body faces above a cut plane, close to the axis
body = np.where(G == 'body')[0]
bverts = np.unique(np.concatenate([F[f] for f in body]))
mouth_y = Vn[LMK['mouth_corner'], 1]
cl_ = bverts[(np.abs(Vn[bverts, 0]) < 1e-4) & (Vn[bverts, 1] > mouth_y - 0.60) & (Vn[bverts, 1] < mouth_y - 0.20)]
cl_ = cl_[Vn[cl_, 2] > Vn[cl_, 2].max() - 0.30]
chin_y = Vn[cl_, 1].min()                                    # menton: lowest front point of the chin on the midline
cut_y = chin_y - CFG['neck_below_chin_dm']
MARGIN = 0.35                                                # extra rows kept below the cut, removed by the bisect
sel = [f for f in body if all(Vn[v, 1] > cut_y - MARGIN for v in F[f]) and all(abs(Vn[v, 0]) < CFG['max_half_width_dm'] for v in F[f])]
# keep only the connected component(s) that touch the head (drops stray shoulder bits)
used = sorted(set(v for f in sel for v in F[f]))
remap = {v: k for k, v in enumerate(used)}
print('head faces', len(sel), 'verts', len(used), 'cut_y', round(cut_y, 3), 'chin_y', round(chin_y, 3))

origin = np.array([0.0, cut_y, Vn[used][np.abs(Vn[used][:, 1] - cut_y) < 0.15][:, 2].mean()])   # neck centre on the cut plane
def conv(P): return M.to_blender(np.asarray(P) - origin)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

def make_object(name, verts, faces, uvs=None, face_uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    if uvs is not None:
        ul = me.uv_layers.new(name='UVMap')
        flat = [uvs[t] for fu in face_uv for t in fu]
        ul.data.foreach_set('uv', np.array(flat, np.float32).ravel())
    for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    return ob

head = make_object('Head', conv(Vn[used]), [[remap[v] for v in F[f]] for f in sel], UV, [FUV[f] for f in sel])
head['orig_index'] = [int(v) for v in used]

PROXIES = {}
def add_proxy(key, rel, name):
    clo = M.load_mhclo(M.PACK + '/' + rel + '.mhclo')
    pv, puv, pf, pfuv, _ = M.load_obj(clo['dir'] + '/' + clo['obj_file'])
    P = M.fit_proxy(clo, Vn)
    assert len(P) == len(pv), (key, len(P), len(pv))
    ob = make_object(name, conv(P), pf, puv if len(puv) else None, pfuv if len(puv) else None)
    ob['mhclo'] = rel
    mat = M.read_mhmat(os.path.normpath(os.path.join(clo['dir'], clo['material'])))
    ob['texture'] = os.path.normpath(os.path.join(os.path.dirname(os.path.join(clo['dir'], clo['material'])), mat['diffuseTexture'][0]))
    PROXIES[key] = ob
    print(key, len(pv), 'verts', sum(len(f) - 2 for f in pf), 'tris')
    return ob
add_proxy('eyes', 'eyes/high-poly/high-poly', 'Eyes')
add_proxy('eyes_low', 'eyes/low-poly/low-poly', 'Eyes_low')
add_proxy('lashes', 'eyelashes/' + CFG['lashes'] + '/' + CFG['lashes'], 'Eyelashes')
add_proxy('brows', 'eyebrows/' + CFG['brows'] + '/' + CFG['brows'], 'Eyebrows')
def thicken_brows(P):
    # scale each brow vertically about its own centre line (thicker brows, as in the reference)
    P = P.copy(); k = CFG.get('brow_thickness', 1.0)
    for sgn in (1, -1):
        m = np.sign(P[:, 0]) == sgn
        # local centre height as a function of x: fit a line z(x)
        A_ = np.stack([P[m, 0], np.ones(m.sum())], 1); co = np.linalg.lstsq(A_, P[m, 2], rcond=None)[0]
        zc = A_ @ co
        P[m, 2] = zc + (P[m, 2] - zc) * k
    return P
_b = PROXIES['brows']; _P = np.array([v.co[:] for v in _b.data.vertices]); _b.data.vertices.foreach_set('co', thicken_brows(_P).astype(np.float32).ravel())
add_proxy('teeth', 'teeth/teeth_base/teeth_base', 'Teeth')
add_proxy('tongue', 'tongue/tongue01/tongue01', 'Tongue')
add_proxy('hair', 'hair/' + CFG['hair'] + '/' + CFG['hair'], 'Hair')
head['texture'] = M.PACK + '/skins/' + CFG['skin'] + '/' + M.read_mhmat(M.PACK + '/skins/' + CFG['skin'] + '/' + CFG['skin'] + '.mhmat')['diffuseTexture'][0]

# ---- measured proportions, in inter-pupil units, to compare with the reference
E = PROXIES['eyes'].data
ev = np.array([v.co[:] for v in E.vertices])
pl = ev[ev[:, 0] > 0].mean(0); pr = ev[ev[:, 0] < 0].mean(0)
ipd = pl[0] - pr[0]
hv = np.array([v.co[:] for v in head.data.vertices])
def L(k): return conv(Vn[[LMK[k]]])[0]
front = hv[(hv[:, 1] < L('cheekbone')[1] + 0.03)]
chin_z = (chin_y - origin[1]) * 0.1
metrics = {
    'ipd_mm': round(ipd * 1000, 1),
    'face_width_cheekbones': round(2 * np.abs(hv[(np.abs(hv[:, 2] - (pl[2] - 0.022)) < 0.008) & (hv[:, 1] < pl[1] + 0.075)][:, 0]).max() / ipd, 3),
    'eye_to_nose_base': round((pl[2] - L('alar')[2]) / ipd, 3),
    'eye_to_mouth': round((pl[2] - L('mouth_corner')[2]) / ipd, 3),
    'eye_to_chin': round((pl[2] - chin_z) / ipd, 3),
    'nose_width': round(2 * abs(L('alar')[0]) / ipd, 3),
    'mouth_width': round(2 * abs(L('mouth_corner')[0]) / ipd, 3),
    'head_height_mm': round((hv[:, 2].max() - chin_z) * 1000, 1),
    'head_width_mm': round((hv[:, 0].max() - hv[:, 0].min()) * 1000, 1),
}
json.dump({'metrics': metrics, 'landmarks': LMK, 'origin_mh': origin.tolist(), 'cut_y': cut_y}, open(W + 'stage1.json', 'w'), indent=1)
print(json.dumps(metrics))
# ---- blend shapes from MakeHuman's expression units (FACS-like), named after ARKit where they match
EXPR = {'jawOpen': ['mouth-open'], 'eyeBlinkLeft': ['eye-left-closure'], 'eyeBlinkRight': ['eye-right-closure'],
        'eyeSquintLeft': ['eye-left-slit'], 'eyeSquintRight': ['eye-right-slit'],
        'eyeWideLeft': ['eye-left-opened-up'], 'eyeWideRight': ['eye-right-opened-up'],
        'browInnerUp': ['eyebrows-left-inner-up', 'eyebrows-right-inner-up'],
        'browDownLeft': ['eyebrows-left-down'], 'browDownRight': ['eyebrows-right-down'],
        'browOuterUpLeft': ['eyebrows-left-extern-up'], 'browOuterUpRight': ['eyebrows-right-extern-up'],
        'mouthSmile': ['mouth-corner-puller'], 'mouthFrown': ['mouth-depression'], 'mouthPucker': ['mouth-pursing'],
        'mouthFunnel': ['mouth-protusion'], 'mouthPress': ['mouth-compression'], 'mouthStretch': ['mouth-retraction'],
        'noseSneer': ['nose-left-elevation', 'nose-right-elevation'], 'mouthUpperUp': ['mouth-elevation']}
UNITS = TGT_EXPR = M.TGT + '/expression/units/' + CFG['expression_set'] + '/'
CLOS = {k: M.load_mhclo(M.PACK + '/' + ob['mhclo'] + '.mhclo') for k, ob in PROXIES.items()}
for ob in [head] + [PROXIES[k] for k in ('lashes', 'brows', 'teeth', 'tongue')]: ob.shape_key_add(name='Basis')
PFIT = {}
for sname, units in EXPR.items():
    Ve = Vn.copy()
    for u in units:
        i, o = M.load_target(UNITS + u + '.target')
        Ve[i] += o
    head.shape_key_add(name=sname).data.foreach_set('co', conv(Ve[used]).astype(np.float32).ravel())
    for k in ('lashes', 'brows', 'teeth', 'tongue'):
        ob = PROXIES[k]
        Pe = conv(M.fit_proxy(CLOS[k], Ve)); P0 = conv(M.fit_proxy(CLOS[k], Vn))
        if k == 'brows': Pe = thicken_brows(Pe); P0 = thicken_brows(P0)
        PFIT.setdefault(k, {})['Basis'] = P0; PFIT[k][sname] = Pe

def kabsch(A, B):
    """Best rigid transform taking points A onto B."""
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, cb - Rm @ ca

def components(ob):
    n = len(ob.data.vertices); par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for e in ob.data.edges:
        a_, b_ = find(e.vertices[0]), find(e.vertices[1])
        if a_ != b_: par[a_] = b_
    return np.array([find(i) for i in range(n)])

from scipy.spatial import cKDTree as _KD
MIN_MOVE = 5e-4                                                     # a proxy gets a key only where it really has to follow the skin
# lashes and brows follow the skin; a right-hand key that came out empty is rebuilt as the mirror of the left one
for k in ('lashes', 'brows'):
    ob = PROXIES[k]; P0 = PFIT[k]['Basis']
    mir = _KD(P0).query(P0 * np.array([-1, 1, 1]))[1]
    for sname in EXPR:
        D = PFIT[k][sname] - P0
        other = sname.replace('Right', 'Left') if 'Right' in sname else sname.replace('Left', 'Right') if 'Left' in sname else None
        if other and other in PFIT[k]:
            Do = (PFIT[k][other] - P0)[mir] * np.array([-1, 1, 1])
            if np.abs(D).max() < 0.5 * np.abs(Do).max(): D = Do
        if np.abs(D).max() > MIN_MOVE:
            ob.shape_key_add(name=sname).data.foreach_set('co', (P0 + D).astype(np.float32).ravel())
# teeth are rigid: the upper arch never moves, the lower arch and the tongue turn with the jaw as one solid piece
T0 = PFIT['teeth']['Basis']; Tj = PFIT['teeth']['jawOpen']; comp = components(PROXIES['teeth'])
low = np.zeros(len(T0), bool)
for c in np.unique(comp):
    m = comp == c
    if np.linalg.norm((Tj - T0)[m], axis=1).mean() > 0.005: low |= m
Rj, tj = kabsch(T0[low], Tj[low])
Tr = T0.copy(); Tr[low] = T0[low] @ Rj.T + tj
PROXIES['teeth'].shape_key_add(name='jawOpen').data.foreach_set('co', Tr.astype(np.float32).ravel())
G0 = PFIT['tongue']['Basis']
PROXIES['tongue'].shape_key_add(name='jawOpen').data.foreach_set('co', (G0 @ Rj.T + tj).astype(np.float32).ravel())
print('teeth: lower verts', int(low.sum()), 'of', len(T0), 'jaw rotation deg', round(float(np.degrees(np.arccos(np.clip((np.trace(Rj) - 1) / 2, -1, 1)))), 1))
for ob in [head] + list(PROXIES.values()):
    if ob.data.shape_keys:
        for kb in ob.data.shape_keys.key_blocks[1:]: kb.value = 0.0
print('shape keys:', {ob.name: len(ob.data.shape_keys.key_blocks) - 1 for ob in [head] + list(PROXIES.values()) if ob.data.shape_keys})

# ---- keep the mouth lining inside the skin in every shape (stacked morphs can push it through the cheeks and chin)
from mathutils.bvhtree import BVHTree
from mathutils import Vector
par = {}
def find(a):
    while par.setdefault(a, a) != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for f in sel:
    r = find(FUV[f][0])
    for t in FUV[f][1:]:
        q = find(t)
        if q != r: par[q] = r
isl = np.array([find(FUV[f][0]) for f in sel])
faces_l = [[remap[v] for v in F[f]] for f in sel]
ids = np.unique(isl)
muv = {k: np.mean([UV[t] for i in np.where(isl == k)[0] for t in FUV[sel[i]]], 0) for k in ids}
lin_id = [k for k in ids if muv[k][0] > 0.7 and muv[k][1] < 0.2][0]          # mouth interior in the MakeHuman layout
sock = [k for k in ids if muv[k][0] < 0.2]                                    # eye-socket interiors
outer_f = [faces_l[i] for i in range(len(sel)) if isl[i] != lin_id and isl[i] not in sock]
lin_v = sorted(set(v for i in np.where(isl == lin_id)[0] for v in faces_l[i]) - set(v for f in outer_f for v in f))
print('islands mean uv', {int(k): muv[k].round(2).tolist() for k in ids}, 'lining', int(lin_id))
kbs = head.data.shape_keys.key_blocks
fixed = 0
for kb in kbs:
    C = np.empty(len(hv) * 3, np.float32); kb.data.foreach_get('co', C); C = C.reshape(-1, 3)
    ctr = C.mean(0)
    keep = []
    for f in outer_f:
        q = C[f]; nrm = np.cross(q[1] - q[0], q[2] - q[0])
        if nrm @ (q.mean(0) - ctr) > 0: keep.append(f)             # outward-facing skin only (not the inner side of the lips)
    bvh = BVHTree.FromPolygons([Vector(c) for c in C], keep)
    for v in lin_v:
        loc, nrm, _, dist = bvh.find_nearest(Vector(C[v]))
        if loc is None or dist > 0.02: continue
        sd = (Vector(C[v]) - loc).dot(nrm)
        if sd > -0.0005:                                          # at or outside the skin: tuck it 1.5 mm under
            C[v] = np.array(loc - nrm * 0.0015); fixed += 1
    kb.data.foreach_set('co', C.ravel())
head.data.vertices.foreach_set('co', np.array([c.co[:] for c in kbs[0].data], np.float32).ravel())
print('mouth lining: %d verts, %d corrections over %d shapes' % (len(lin_v), fixed, len(kbs)))

# ---- clean planar neck cut (bmesh keeps UVs and shape-key layers through the bisect)
import bmesh
bm = bmesh.new(); bm.from_mesh(head.data)
bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_inner=True)
# drop pieces not connected to the face (stray shoulder bits)
bm.verts.ensure_lookup_table()
seed = max(bm.verts, key=lambda v: v.co.z); seen = {seed}; stack = [seed]
while stack:
    v = stack.pop()
    for e in v.link_edges:
        o = e.other_vert(v)
        if o not in seen: seen.add(o); stack.append(o)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v not in seen], context='VERTS')
bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
print('head after cut: verts', len(bm.verts), 'faces', len(bm.faces), 'tris', sum(len(f.verts) - 2 for f in bm.faces),
      'sides', {n: sum(1 for f in bm.faces if len(f.verts) == n) for n in (3, 4, 5, 6)}, 'boundary edges', sum(e.is_boundary for e in bm.edges))
bm.to_mesh(head.data); bm.free()
np.save(W + 'Vn.npy', Vn)
bpy.ops.wm.save_as_mainfile(filepath=W + 'stage1.blend')
