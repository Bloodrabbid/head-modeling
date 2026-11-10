import os
"""v2, stage 1: character from the MakeHuman base mesh (CC0) shaped toward the reference,
head + neck cut out, proxies fitted (eyes, lashes, brows, teeth, tongue, hair). Saves work/stage1.blend."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy
import mhlib as M

W = P.WORK + ''
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
ESC = CFG.get('eye_scale', 1.0)
if ESC != 1.0:
    for ek in ('eyes', 'eyes_low'):
        ob = PROXIES[ek]; E0 = np.array([v.co[:] for v in ob.data.vertices])
        for sgn in (1, -1):
            m = np.sign(E0[:, 0]) == sgn; c = E0[m].mean(0); front = c.copy(); front[1] = E0[m][:, 1].min()
            pivot = c + (front - c) * 0.55
            E0[m] = pivot + (E0[m] - pivot) * ESC
        ob.data.vertices.foreach_set('co', E0.astype(np.float32).ravel())
    print('eyeballs scaled', ESC)
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
# brows sit higher above the eyes in the reference than the brow ridge puts them: slide the brow cards up the forehead,
# keeping each vertex's distance from the skin, and carry the same shift into every expression
from mathutils.bvhtree import BVHTree as _BVH
from mathutils import Vector as _Vec
_P0b = np.array([v.co[:] for v in _b.data.vertices]); BROW_DELTA = np.zeros_like(_P0b)
_rz = CFG.get('brow_raise_mm', 0) / 1000.0
if _rz:
    _bv = _BVH.FromObject(head, bpy.context.evaluated_depsgraph_get())
    for _i, _p in enumerate(_P0b):
        _l, _n, _, _ = _bv.find_nearest(_Vec(_p)); _d0 = (_Vec(_p) - _l).dot(_n)
        _xr = abs(_p[0]) / max(1e-6, np.abs(_P0b[:, 0]).max()); _rzi = _rz + CFG.get('brow_inner_raise_mm', 0) / 1000.0 * (1 - _xr) ** 1.5
        _l2, _n2, _, _ = _bv.find_nearest(_Vec(_p) + _Vec((0, 0, _rzi)))
        BROW_DELTA[_i] = np.array(_l2 + _n2 * _d0) - _p
    _b.data.vertices.foreach_set('co', (_P0b + BROW_DELTA).astype(np.float32).ravel())
    print('brows raised', round(float(BROW_DELTA[:, 2].mean() * 1000), 2), 'mm up,', round(float(BROW_DELTA[:, 1].mean() * 1000), 2), 'mm back')
add_proxy('teeth', 'teeth/teeth_base/teeth_base', 'Teeth')
add_proxy('tongue', 'tongue/tongue01/tongue01', 'Tongue')
if CFG['hair'] != 'groom': add_proxy('hair', 'hair/' + CFG['hair'] + '/' + CFG['hair'], 'Hair')
# ---- loose strands framing the face (the reference wears its hair pulled back with strands falling to the jaw):
# ribbon cards grown from the hairline down the sides of the face, kept a few millimetres off the skin
def hair_cards(n_side=16, n_ear=9, n_fore=3, n_scalp=150, seed=7):
    from mathutils.bvhtree import BVHTree as _B
    from mathutils import Vector as _V
    rng = np.random.default_rng(seed)
    bvh = _B.FromObject(head, bpy.context.evaluated_depsgraph_get())
    HVb = np.array([v.co[:] for v in head.data.vertices])
    eye_z = np.array([v.co[:] for v in PROXIES['eyes'].data.vertices])[:, 2].mean()
    brow = np.array([0.0, 0.0, eye_z + 0.025]); chinb = conv(Vn[[LMK['chin']]])[0]
    zc = eye_z + 0.03; _sl = HVb[np.abs(HVb[:, 2] - zc) < 0.01][:, 1]; yc = 0.5 * (_sl.min() + _sl.max())
    def surf(phi, z, extra=0.0):
        d = np.array([np.sin(phi), -np.cos(phi), 0.0]); far = np.array([0, yc, z]) + d * 0.4
        hit = bvh.ray_cast(_V(far), _V(-d))
        if hit[0] is None: return None
        return np.array(hit[0]) + np.array(hit[1]) * extra
    verts, faces, uvs = [], [], []
    bvh_cap = _B.FromObject(PROXIES['hair'], bpy.context.evaluated_depsgraph_get())
    def card(S, E, ctrl1, ctrl2, w0, d0, d1, u0, nseg=12, over_cap=0.0):
        base = len(verts)
        for k in range(nseg + 1):
            t = k / nseg
            p = (1 - t) ** 3 * S + 3 * (1 - t) ** 2 * t * ctrl1 + 3 * (1 - t) * t * t * ctrl2 + t ** 3 * E
            loc, nrm, _, dist = bvh.find_nearest(_V(p)); loc, nrm = np.array(loc), np.array(nrm)
            dmin = d0 + (d1 - d0) * t
            if (p - loc) @ nrm < dmin: p = loc + nrm * dmin
            if over_cap > 0:                                       # scalp strands lie on top of the hair cap
                lc, nc, _, dc = bvh_cap.find_nearest(_V(p))
                if lc is not None and dc < 0.03:
                    lc, nc = np.array(lc), np.array(nc)
                    if (p - lc) @ nc < over_cap: p = lc + nc * over_cap
            tn = (3 * (1 - t) ** 2 * (ctrl1 - S) + 6 * (1 - t) * t * (ctrl2 - ctrl1) + 3 * t * t * (E - ctrl2)); tn /= np.linalg.norm(tn) + 1e-12
            bn = np.cross(tn, nrm); bn /= np.linalg.norm(bn) + 1e-12
            w = w0 * (1 - 0.55 * t)
            verts.extend([p - bn * w / 2, p + bn * w / 2]); uvs.extend([(u0, 1 - t), (u0 + 0.24, 1 - t)])
        for k in range(nseg):
            a = base + 2 * k; faces.append((a, a + 1, a + 3, a + 2))
    z_h = eye_z + 0.072
    print('hair cards debug: brow', brow.round(3), 'chin', chinb.round(3), 'yc', round(float(yc), 3), 'sample', surf(np.radians(50), z_h), surf(np.radians(50), chinb[2] + 0.02))
    for sgn in (1, -1):
        for i in range(n_side):                                    # temple strands falling past the cheek to the jaw
            phi = sgn * np.radians(rng.uniform(32, 68)); z0 = z_h + rng.uniform(-0.01, 0.012)
            S = surf(phi, z0, 0.002); E = surf(phi + sgn * np.radians(rng.uniform(4, 16)), chinb[2] + rng.uniform(0.01, 0.045), rng.uniform(0.008, 0.016))
            if S is None or E is None: continue
            c1 = S + (E - S) * 0.35 + np.array([sgn * 0.012, -0.004, 0.0]); c2 = E + np.array([sgn * 0.004, -0.006, 0.02])
            card(S, E, c1, c2, rng.uniform(0.006, 0.011), 0.002, rng.uniform(0.006, 0.012), 0.5 + rng.integers(0, 2) * 0.25)
        for i in range(n_ear):                                     # strands over the ear
            phi = sgn * np.radians(rng.uniform(72, 104)); z0 = z_h + rng.uniform(-0.02, 0.0)
            S = surf(phi, z0, 0.002); E = surf(phi + sgn * np.radians(rng.uniform(-5, 8)), chinb[2] + rng.uniform(0.02, 0.06), rng.uniform(0.008, 0.014))
            if S is None or E is None: continue
            card(S, E, S + (E - S) * 0.3 + np.array([sgn * 0.014, 0, 0]), E + np.array([sgn * 0.006, 0, 0.02]), rng.uniform(0.008, 0.013), 0.003, 0.010, 0.5 + rng.integers(0, 2) * 0.25)
    for i in range(n_fore):                                        # a few wisps across the forehead
        sgn = 1 if i % 2 == 0 else -1
        phi = sgn * np.radians(rng.uniform(5, 25)); S = surf(phi, z_h + 0.008, 0.002)
        E = surf(phi + sgn * np.radians(rng.uniform(20, 32)), brow[2] + rng.uniform(0.012, 0.03), 0.0025)
        if S is None or E is None: continue
        card(S, E, S + (E - S) * 0.4, E + np.array([0, 0, 0.012]), rng.uniform(0.004, 0.007), 0.0015, 0.0025, 0.5 + rng.integers(0, 2) * 0.25, nseg=8)
    # combed-back strands over the scalp, from the hairline to where the ponytail is tied
    HP = np.array([v.co[:] for v in PROXIES['hair'].data.vertices])
    yb = HVb[:, 1].max(); tie_m = (HP[:, 1] > yb - 0.03) & (HP[:, 2] > eye_z - 0.01) & (HP[:, 2] < eye_z + 0.06) & (np.abs(HP[:, 0]) < 0.03)
    tie = HP[tie_m].mean(0) if tie_m.sum() > 5 else np.array([0.0, yb, eye_z + 0.03])
    ZK = ([0, 60, 90, 120, 180], [z_h, z_h - 0.012, eye_z + 0.035, eye_z - 0.015, eye_z - 0.04])
    for i in range(n_scalp):
        deg = rng.uniform(-180, 180); phi = np.radians(deg)
        z0 = np.interp(abs(deg), *ZK) + rng.uniform(-0.004, 0.008)
        S = surf(phi, z0, 0.003)
        if S is None: continue
        E = tie + np.array([rng.uniform(-0.012, 0.012), rng.uniform(-0.01, 0.0), rng.uniform(-0.012, 0.012)])
        if abs(deg) < 110:
            top = np.array([S[0] * 0.8, 0.5 * (S[1] + E[1]), max(S[2], E[2]) + 0.06 - 0.03 * abs(np.sin(phi))])
            c1, c2 = S + (top - S) * 0.9, E + (top - E) * 0.6
        else:                                                      # behind the ears and the nape: brushed up to the tie
            mid = 0.5 * (S + E); out = np.array([mid[0], mid[1] + 0.02, mid[2]]) - np.array([0, tie[1] - 0.04, mid[2]]); out[2] = 0
            c1, c2 = S + (E - S) * 0.3 + out * 0.3, E + (S - E) * 0.3 + out * 0.3
        card(S, E, c1, c2, rng.uniform(0.012, 0.020), 0.004, 0.006, rng.integers(0, 2) * 0.25, nseg=16, over_cap=0.0015)
    me = bpy.data.meshes.new('HairCards'); me.from_pydata([tuple(v) for v in verts], [], faces); me.update()
    ul = me.uv_layers.new(name='UVMap'); ul.data.foreach_set('uv', np.array([uvs[i] for f in faces for i in f], np.float32).ravel())
    for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new('HairCards', me); sc.collection.objects.link(ob)
    print('hair cards:', len(faces) // 1, 'quads,', len(verts), 'verts')
    return ob
# ---- groomed bob from cards (no MakeHuman hair): a dark cap over the scalp + layered strand cards from a side part,
# falling to about chin length, a few loose strands at the temples
def groom_bob(seed=11):
    from mathutils.bvhtree import BVHTree as _B
    from mathutils import Vector as _V
    rng = np.random.default_rng(seed)
    dg = bpy.context.evaluated_depsgraph_get(); bvh = _B.FromObject(head, dg)
    HVb = np.array([v.co[:] for v in head.data.vertices]); HN = np.array([v.normal[:] for v in head.data.vertices])
    eye_z = np.array([v.co[:] for v in PROXIES['eyes'].data.vertices])[:, 2].mean()
    chinb = conv(Vn[[LMK['chin']]])[0]
    _sl = HVb[np.abs(HVb[:, 2] - (eye_z + 0.03)) < 0.01][:, 1]; yc = 0.5 * (_sl.min() + _sl.max())
    top_z = HVb[:, 2].max()
    HL = CFG.get('hairline', [[0, 0.070], [45, 0.058], [80, 0.030], [110, -0.010], [180, -0.055]])   # degrees from front -> z above the eyes
    def hairline_z(phi_deg): return eye_z + np.interp(abs(phi_deg), [p for p, _ in HL], [z for _, z in HL])
    def azim(P): return np.degrees(np.arctan2(P[..., 0], -(P[..., 1] - yc)))
    # cap: head faces above the hairline, pushed 1.6 mm out along the vertex normals
    cap_f = []
    for p in head.data.polygons:
        c = np.array(p.center)
        if c[2] > hairline_z(azim(c)) + 0.002: cap_f.append(list(p.vertices))
    used_v = sorted(set(v for f in cap_f for v in f)); rm = {v: i for i, v in enumerate(used_v)}
    CV = HVb[used_v] + HN[used_v] * 0.0016
    me = bpy.data.meshes.new('Hair'); me.from_pydata([tuple(v) for v in CV], [], [[rm[v] for v in f] for f in cap_f]); me.update()
    az = azim(CV); uvc = np.c_[(az + 180) / 360, (CV[:, 2] - eye_z + 0.08) / (top_z - eye_z + 0.08)]
    ul = me.uv_layers.new(name='UVMap'); ul.data.foreach_set('uv', np.array([uvc[rm[v]] for f in cap_f for v in f], np.float32).ravel())
    for p in me.polygons: p.use_smooth = True
    cap = bpy.data.objects.new('Hair', me); sc.collection.objects.link(cap); cap['procedural'] = 1
    bcap = _B.FromObject(cap, bpy.context.evaluated_depsgraph_get())
    def surf(phi, z, extra=0.0):
        d = np.array([np.sin(phi), -np.cos(phi), 0.0]); far = np.array([0, yc, z]) + d * 0.4
        hit = bvh.ray_cast(_V(far), _V(-d))
        return None if hit[0] is None else np.array(hit[0]) + np.array(hit[1]) * extra
    verts, faces, uvs = [], [], []
    def card(pts, w0, d0, d1, u0, nseg=16, taper=0.5):
        base = len(verts)
        P = np.array(pts)
        for k in range(nseg + 1):
            t = k / nseg
            # Catmull-Rom through the control points
            seg = min(int(t * (len(P) - 1)), len(P) - 2); lt = t * (len(P) - 1) - seg
            p0, p1, p2, p3 = P[max(seg - 1, 0)], P[seg], P[seg + 1], P[min(seg + 2, len(P) - 1)]
            p = 0.5 * ((2 * p1) + (-p0 + p2) * lt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * lt ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * lt ** 3)
            dmin = d0 + (d1 - d0) * t
            for _ in range(2):
                loc, nrm, _, _ = bvh.find_nearest(_V(p)); loc, nrm = np.array(loc), np.array(nrm)
                if (p - loc) @ nrm < dmin: p = loc + nrm * dmin
                lc, nc, _, dc = bcap.find_nearest(_V(p))
                if lc is not None and dc < 0.02:
                    lc, nc = np.array(lc), np.array(nc)
                    if (p - lc) @ nc < 0.0012: p = lc + nc * 0.0012
            verts.append(p); uvs.append(t)
        Q = np.array(verts[base:])
        out = []
        for k in range(nseg + 1):
            t = k / nseg; q = Q[k]
            tn = Q[min(k + 1, nseg)] - Q[max(k - 1, 0)]; tn /= np.linalg.norm(tn) + 1e-12
            loc, nrm, _, _ = bvh.find_nearest(_V(q)); nrm = np.array(nrm)
            bn = np.cross(tn, nrm); bn /= np.linalg.norm(bn) + 1e-12
            w = w0 * (1 - taper * t) * (0.45 + 0.55 * min(1.0, t / 0.12))      # narrow at the root, no flat tiles at the part
            out += [q - bn * w / 2, q + bn * w / 2]
        del verts[base:]; del uvs[base:]
        for k in range(nseg + 1):
            verts.extend(out[2 * k: 2 * k + 2]); uvs.extend([(u0, 1 - k / nseg), (u0 + 0.24, 1 - k / nseg)])
        for k in range(nseg):
            a = base + 2 * k; faces.append((a, a + 1, a + 3, a + 2))
    part_x = CFG.get('part_x', -0.010)
    front = surf(0.0, hairline_z(0) - 0.002, 0.003); crown_y = yc + 0.045
    n_cards = CFG.get('groom_cards', 260)
    for i in range(n_cards):
        t = rng.uniform(0, 1) ** 0.8                               # position along the part, front to crown
        sgn = 1 if rng.random() < 0.5 else -1
        root = np.array([part_x + sgn * rng.uniform(0.002, 0.012), front[1] + (crown_y - front[1]) * t, 0.0])
        hit = bvh.ray_cast(_V((root[0], root[1], top_z + 0.2)), _V((0, 0, -1)))
        if hit[0] is None: continue
        S = np.array(hit[0]) + np.array(hit[1]) * 0.003
        phi_e = sgn * (rng.uniform(38, 60) + t * rng.uniform(95, 135)); phi_e = float(np.clip(phi_e, -178, 178))
        frame = abs(phi_e) < 75
        z_e = chinb[2] + (rng.uniform(-0.012, 0.04) if frame else rng.uniform(-0.01, 0.05))
        M1 = surf(np.radians(phi_e * 0.55), max(hairline_z(phi_e * 0.55) + 0.03, eye_z + 0.075), 0.006)
        M2 = surf(np.radians(phi_e * 0.92), eye_z + rng.uniform(0.0, 0.03), rng.uniform(0.008, 0.014))
        E = surf(np.radians(phi_e), z_e, rng.uniform(0.014, 0.026))
        if M1 is None or M2 is None or E is None: continue
        layer = rng.random() < 0.55
        card([S, M1, M2, E], rng.uniform(0.014, 0.022) if layer else rng.uniform(0.009, 0.015), 0.003, 0.010 if layer else 0.016,
             rng.integers(0, 2) * 0.25 if layer else 0.5 + rng.integers(0, 2) * 0.25, nseg=16, taper=0.45)
    for i in range(CFG.get('groom_part', 40)):                     # short cards lying across the part so the cap never shows there
        t = rng.uniform(0, 1); sgn = 1 if i % 2 else -1
        rp = np.array([part_x - sgn * 0.006, front[1] + (crown_y - front[1]) * t])
        hs = bvh.ray_cast(_V((rp[0], rp[1], top_z + 0.2)), _V((0, 0, -1)))
        if hs[0] is None: continue
        S = np.array(hs[0]) + np.array(hs[1]) * 0.004
        M_ = surf(np.radians(sgn * (35 + 90 * t)), eye_z + 0.075, 0.007)
        if M_ is None: continue
        card([S, 0.5 * (S + M_) + np.array([0, 0, 0.012]), M_], rng.uniform(0.016, 0.024), 0.004, 0.007, rng.integers(0, 2) * 0.25, nseg=10, taper=0.3)
    for i in range(CFG.get('groom_nape', 40)):                     # nape: strands from just above the hairline down past it
        phi = rng.uniform(110, 250); phi = phi if phi <= 180 else phi - 360
        S = surf(np.radians(phi), hairline_z(phi) + rng.uniform(0.01, 0.03), 0.003)
        E = surf(np.radians(phi), chinb[2] + rng.uniform(-0.01, 0.02), rng.uniform(0.008, 0.014))
        if S is None or E is None: continue
        card([S, 0.5 * (S + E) + (E - S) * 0.1, E], rng.uniform(0.014, 0.02), 0.004, 0.010, rng.integers(0, 2) * 0.25, nseg=10, taper=0.4)
    crown = np.array([part_x, crown_y + 0.01, 0.0])                # the back: strands from the crown down to the nape
    hitc = bvh.ray_cast(_V((crown[0], crown[1], top_z + 0.2)), _V((0, 0, -1)))
    if hitc[0] is not None:
        C0 = np.array(hitc[0]) + np.array(hitc[1]) * 0.003
        for i in range(CFG.get('groom_back', 90)):
            phi_e = rng.uniform(95, 180) * (1 if rng.random() < 0.5 else -1)
            Sx = C0 + np.array([rng.uniform(-0.012, 0.012), rng.uniform(-0.01, 0.01), 0])
            hs = bvh.ray_cast(_V((Sx[0], Sx[1], top_z + 0.2)), _V((0, 0, -1)))
            if hs[0] is None: continue
            S = np.array(hs[0]) + np.array(hs[1]) * 0.003
            M1 = surf(np.radians(phi_e), eye_z + 0.06, 0.006); E = surf(np.radians(phi_e), chinb[2] + rng.uniform(-0.005, 0.05), rng.uniform(0.012, 0.022))
            if M1 is None or E is None: continue
            card([S, M1, E], rng.uniform(0.016, 0.024), 0.003, 0.012, rng.integers(0, 2) * 0.25, nseg=14, taper=0.4)
    for sgn in (1, -1):                                            # loose strands framing the face
        for i in range(CFG.get('groom_loose', 10)):
            phi = sgn * rng.uniform(28, 55); S = surf(np.radians(phi), hairline_z(phi) + rng.uniform(0.0, 0.01), 0.003)
            E = surf(np.radians(phi + sgn * rng.uniform(5, 15)), chinb[2] + rng.uniform(0.0, 0.035), rng.uniform(0.006, 0.012))
            M_ = surf(np.radians(phi + sgn * 4), eye_z + rng.uniform(-0.01, 0.02), rng.uniform(0.004, 0.008))
            if S is None or E is None or M_ is None: continue
            card([S, M_, E], rng.uniform(0.004, 0.008), 0.002, 0.008, 0.5 + rng.integers(0, 2) * 0.25, nseg=14, taper=0.6)
    me = bpy.data.meshes.new('HairCards'); me.from_pydata([tuple(v) for v in verts], [], faces); me.update()
    ul = me.uv_layers.new(name='UVMap'); ul.data.foreach_set('uv', np.array([uvs[i] for f in faces for i in f], np.float32).ravel())
    for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new('HairCards', me); sc.collection.objects.link(ob)
    print('groom: cap', len(cap_f), 'faces; cards', len(faces), 'quads')
    return ob
if CFG['hair'] == 'groom': HAIRCARDS = groom_bob()
elif CFG.get('hair_cards', True): HAIRCARDS = hair_cards()
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
UNITS = TGT_EXPR = M.TGT + '/expression/units/' + CFG['expression_set'] + '/'
CLOS = {k: M.load_mhclo(M.PACK + '/' + ob['mhclo'] + '.mhclo') for k, ob in PROXIES.items()}
for ob in [head] + [PROXIES[k] for k in ('lashes', 'brows', 'teeth', 'tongue')]: ob.shape_key_add(name='Basis')
# ---- the 52 ARKit blend shapes, in MakeHuman space (dm, Y up, Z forward, +X = character's left).
# Most come from MakeHuman's expression units (split into left/right with a soft mask where ARKit wants sides);
# the ones MakeHuman has no unit for are built from small displacement fields around landmarks.
def unit(name):
    D = np.zeros_like(Vn); i, o = M.load_target(UNITS + name + '.target'); D[i] += o; return D
def sstep_(a, b, v): t = np.clip((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
WL = sstep_(-0.08, 0.08, Vn[:, 0])[:, None]; WR = 1 - WL
def side(D, sgn): return D * (WL if sgn > 0 else WR)
def gauss(c, sig): d = Vn - np.asarray(c); return np.exp(-(d * d).sum(1) / (2 * sig * sig))[:, None]
def vnormals(V):
    N = np.zeros_like(V)
    for f in F:
        if len(f) < 3: continue
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]); N[f] += n
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
NRM = vnormals(Vn)
def tw_mag(name):
    D = np.zeros(len(Vn)); i, o = M.load_target(M.TGT + '/' + name + '.target'); n = np.linalg.norm(o, axis=1); D[i] = n / n.max(); return D[:, None]
UPW = tw_mag('mouth/mouth-upperlip-volume-incr'); LOW = tw_mag('mouth/mouth-lowerlip-volume-incr')
D_OPEN = unit('mouth-open'); jm = np.linalg.norm(D_OPEN, axis=1); WJ = np.clip(jm / np.percentile(jm[jm > 0], 95) * 1.1, 0, 1)[:, None]
MC = Vn[LMK['mouth_corner']]; CB = Vn[LMK['cheekbone']]
def cheek_puff():
    D = np.zeros_like(Vn)
    for sgn in (1, -1):
        D += NRM * 0.05 * gauss((sgn * (abs(MC[0]) + 0.15), MC[1] + 0.03, MC[2] - 0.12), 0.22)
    return D * (1 - UPW - LOW).clip(0, 1)
MCC = np.array([0.0, MC[1], MC[2] - 0.02])
from scipy.spatial import cKDTree as _KDm
MIRROR = _KDm(V0 * np.array([-1, 1, 1])).query(V0)[1]                       # vertex -> its mirror twin (base mesh is symmetric)
def mirror(D): return D[MIRROR] * np.array([-1, 1, 1])
import scipy.sparse as _sp
_rows, _cols = [], []
for f in F:
    for k_ in range(len(f)):
        _rows += [f[k_], f[(k_ + 1) % len(f)]]; _cols += [f[(k_ + 1) % len(f)], f[k_]]
ADJ = _sp.csr_matrix((np.ones(len(_rows)), (_rows, _cols)), shape=(len(Vn), len(Vn))); ADJ.data[:] = 1
_deg = np.asarray(ADJ.sum(1)).ravel(); _deg[_deg == 0] = 1; ADJN = _sp.diags(1 / _deg) @ ADJ
def smooth(D, it=4):
    for _ in range(it): D = 0.5 * D + 0.5 * (ADJN @ D)
    return D
CHIN_Y = Vn[LMK['chin'], 1]
W_CLOSE = smooth(sstep_(CHIN_Y + 0.05, MC[1] - 0.13, Vn[:, 1])[:, None], 4)   # 1 over the whole lower lip, 0 toward the chin
CR = 0.05                                                                         # cheek raise, dm
def cheek_raise(sgn): return gauss((sgn * abs(CB[0]) * 0.85, CB[1] - 0.05, CB[2]), 0.17) * np.array([0, CR, CR * 0.3]) * (1 - UPW - LOW).clip(0, 1)
LIPZ = gauss(MCC, 0.16)                                                           # lips move together so the seal holds
def jaw_shift(d): return (WJ * (1 - LIPZ) + 0.6 * LIPZ * (WJ > 0.02)) * np.asarray(d)
def stretch(sgn):
    D = side(unit('mouth-retraction'), sgn); D[:, 1] = np.minimum(D[:, 1], 0)      # no upward pull on the corners
    return D + gauss(MC * np.array([sgn, 1, 1]), 0.12) * np.array([sgn * 0.03, -0.03, -0.005])
EXPR = {
    'jawOpen': D_OPEN,
    'eyeBlinkLeft': unit('eye-left-closure'), 'eyeBlinkRight': unit('eye-right-closure'),
    'eyeSquintLeft': 0.6 * unit('eye-left-slit') + 0.8 * cheek_raise(1), 'eyeSquintRight': 0.6 * unit('eye-right-slit') + 0.8 * cheek_raise(-1),
    'eyeWideLeft': unit('eye-left-opened-up'), 'eyeWideRight': unit('eye-right-opened-up'),
    'eyeLookUpLeft': 0.3 * unit('eye-left-opened-up'), 'eyeLookUpRight': 0.3 * unit('eye-right-opened-up'),
    'eyeLookDownLeft': 0.3 * unit('eye-left-closure'), 'eyeLookDownRight': 0.3 * unit('eye-right-closure'),
    'eyeLookInLeft': 0 * D_OPEN, 'eyeLookInRight': 0 * D_OPEN, 'eyeLookOutLeft': 0 * D_OPEN, 'eyeLookOutRight': 0 * D_OPEN,
    'browInnerUp': unit('eyebrows-left-inner-up') + unit('eyebrows-right-inner-up'),
    'browDownLeft': unit('eyebrows-left-down'), 'browDownRight': unit('eyebrows-right-down'),
    'browOuterUpLeft': unit('eyebrows-left-extern-up'), 'browOuterUpRight': unit('eyebrows-right-extern-up'),
    'cheekSquintLeft': cheek_raise(1), 'cheekSquintRight': cheek_raise(-1), 'cheekPuff': cheek_puff(),
    'noseSneerLeft': unit('nose-left-elevation'), 'noseSneerRight': unit('nose-right-elevation'),
    'mouthSmileLeft': side(unit('mouth-corner-puller'), 1) + 0.9 * cheek_raise(1), 'mouthSmileRight': side(unit('mouth-corner-puller'), -1) + 0.9 * cheek_raise(-1),
    'mouthFrownLeft': side(unit('mouth-depression'), 1), 'mouthFrownRight': side(unit('mouth-depression'), -1),
    'mouthStretchLeft': stretch(1), 'mouthStretchRight': stretch(-1),
    'mouthPressLeft': side(unit('mouth-compression'), 1), 'mouthPressRight': side(unit('mouth-compression'), -1),
    'mouthUpperUpLeft': side(unit('mouth-elevation'), 1), 'mouthUpperUpRight': side(unit('mouth-elevation'), -1),
    'mouthLowerDownLeft': side(unit('mouth-depression-retraction'), 1), 'mouthLowerDownRight': side(unit('mouth-depression-retraction'), -1),
    'mouthDimpleLeft': gauss(MC * np.array([1, 1, 1]), 0.08) * np.array([0.02, 0, -0.02]),
    'mouthDimpleRight': gauss(MC * np.array([-1, 1, 1]), 0.08) * np.array([-0.02, 0, -0.02]),
    'mouthPucker': 0.7 * unit('mouth-pursing'), 'mouthFunnel': unit('mouth-protusion'),
    'mouthRollUpper': UPW * np.array([0, -0.012, -0.03]), 'mouthRollLower': LOW * np.array([0, 0.012, -0.03]),
    'mouthShrugUpper': UPW * np.array([0, 0.015, 0.015]), 'mouthShrugLower': unit('mouth-eversion'),
    'mouthClose': -D_OPEN * W_CLOSE,
    'mouthLeft': gauss(MCC, 0.20) * np.array([0.06, 0, 0]), 'mouthRight': gauss(MCC, 0.20) * np.array([-0.06, 0, 0]),
    'jawForward': jaw_shift([0, 0, 0.045]), 'jawLeft': jaw_shift([0.07, 0, 0]), 'jawRight': jaw_shift([-0.07, 0, 0]),
}
for k_ in ('cheekPuff', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthClose',
           'mouthLeft', 'mouthRight', 'jawForward', 'jawLeft', 'jawRight', 'cheekSquintLeft', 'cheekSquintRight'):
    EXPR[k_] = smooth(EXPR[k_], 3)
for k_ in list(EXPR):
    if k_.endswith('Left') and k_.replace('Left', 'Right') in EXPR and not k_.startswith(('eyeLookIn', 'eyeLookOut')):
        EXPR[k_.replace('Left', 'Right')] = mirror(EXPR[k_])
EXPR['mouthRight'] = mirror(EXPR['mouthLeft']); EXPR['jawRight'] = mirror(EXPR['jawLeft'])
print('blend shapes defined:', len(EXPR) + 1, '(+ tongueOut on the tongue); right side mirrored from left')
PFIT = {}
for sname, D in EXPR.items():
    Ve = Vn + D
    head.shape_key_add(name=sname).data.foreach_set('co', conv(Ve[used]).astype(np.float32).ravel())
    for k in ('lashes', 'brows', 'teeth', 'tongue'):
        ob = PROXIES[k]
        Pe = conv(M.fit_proxy(CLOS[k], Ve)); P0 = conv(M.fit_proxy(CLOS[k], Vn))
        if k == 'brows': Pe = thicken_brows(Pe) + BROW_DELTA; P0 = thicken_brows(P0) + BROW_DELTA
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
MIN_MOVE = 2e-4                                                     # a proxy gets a key only where it really has to follow the skin
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
def mh_vec(d): return (np.asarray(d) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T) * 0.1   # MakeHuman delta -> Blender metres
for kn, d in (('jawForward', (0, 0, 0.045)), ('jawLeft', (0.07, 0, 0)), ('jawRight', (-0.07, 0, 0))):
    Tt = T0.copy(); Tt[low] += mh_vec(d)
    PROXIES['teeth'].shape_key_add(name=kn).data.foreach_set('co', Tt.astype(np.float32).ravel())
    PROXIES['tongue'].shape_key_add(name=kn).data.foreach_set('co', (G0 + mh_vec(d)).astype(np.float32).ravel())
PROXIES['tongue'].shape_key_add(name='tongueOut').data.foreach_set('co', (G0 + np.array([0, -0.018, -0.004])).astype(np.float32).ravel())
# eyes: ARKit gaze keys rotate each eyeball about its own centre (up 25, down 30, in/out 30 degrees)
import math as _m
def sphere_fit(P):
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Q = P[keep]; A_ = np.c_[2 * Q, np.ones(len(Q))]; sol = np.linalg.lstsq(A_, (Q * Q).sum(1), rcond=None)[0]
        c = sol[:3]; r = np.sqrt(sol[3] + c @ c); res = np.abs(np.linalg.norm(P - c, axis=1) - r)
        keep = res <= np.quantile(res, 0.8)
    return c, r
_EH = np.array([v.co[:] for v in PROXIES['eyes'].data.vertices])
EYE_CENTERS = {}
for sgn, sd in ((1, 'Left'), (-1, 'Right')):
    c_, r_ = sphere_fit(_EH[np.sign(_EH[:, 0]) == sgn]); EYE_CENTERS[sd] = c_
    print('eye', sd, 'sphere centre', c_.round(4), 'radius mm', round(float(r_ * 1000), 2), 'vertex mean', _EH[np.sign(_EH[:, 0]) == sgn].mean(0).round(4))
head['eye_centers'] = [float(x) for x in np.r_[EYE_CENTERS['Left'], EYE_CENTERS['Right']]]
for ek in ('eyes', 'eyes_low'):
    ob = PROXIES[ek]; ob.shape_key_add(name='Basis')
    E0 = np.array([v.co[:] for v in ob.data.vertices])
    for sgn, sd in ((1, 'Left'), (-1, 'Right')):
        m = np.sign(E0[:, 0]) == sgn; c = EYE_CENTERS[sd]
        for kn, axis, ang in (('Up', 'x', -25), ('Down', 'x', 30), ('In', 'z', -30 * sgn), ('Out', 'z', 30 * sgn)):
            a_ = _m.radians(ang); ca, sa = _m.cos(a_), _m.sin(a_)
            Rr = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]) if axis == 'x' else np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
            Ek = E0.copy(); Ek[m] = (E0[m] - c) @ Rr.T + c
            ob.shape_key_add(name=f'eyeLook{kn}{sd}').data.foreach_set('co', Ek.astype(np.float32).ravel())
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
_mcb = conv(Vn[[LMK['mouth_corner']]])[0]
_hb = np.array([c.co[:] for c in head.data.shape_keys.key_blocks[0].data])
lin_v = [v for v in lin_v if _hb[v, 2] < _mcb[2] + 0.010]                   # mouth only: nostril interiors stay put
print('islands mean uv', {int(k): muv[k].round(2).tolist() for k in ids}, 'lining', int(lin_id))
kbs = head.data.shape_keys.key_blocks
# ---- close the mouth corners: the inner lip contour (where skin turns into the mouth interior) gapes near the corners
CC = CFG.get('mouth_corner_close', 0.0)
if CC > 0:
    edge_v = sorted(set(v for i in np.where(isl == lin_id)[0] for v in faces_l[i]) & set(v for f in outer_f for v in f))
    es = set(edge_v); adj = {v: set() for v in edge_v}
    for f in [faces_l[i] for i in range(len(sel))]:
        for k_ in range(len(f)):
            a_, b_ = f[k_], f[(k_ + 1) % len(f)]
            if a_ in es and b_ in es: adj[a_].add(b_); adj[b_].add(a_)
    B0 = np.array([kb.co[:] for kb in kbs[0].data])
    # walk the contour: start at the right-most vertex and follow neighbours
    start = max(edge_v, key=lambda v: B0[v, 0]); loop = [start]; prev = None; cur = start
    while True:
        nxt = [n for n in adj[cur] if n != prev and (n not in loop or (n == start and len(loop) > 3))]
        if not nxt: break
        n = min(nxt, key=lambda q: np.linalg.norm(B0[q] - B0[cur]))
        if n == start: break
        loop.append(n); prev, cur = cur, n
    if len(loop) < 8:
        print('mouth corners: contour walk failed, skipped', len(loop))
    else:
        L = np.array(loop); xs = B0[L, 0]
        iR = 0; iL = int(np.argmin(xs))
        pA = L[:iL + 1]; pB = np.concatenate([L[iL:], L[:1]])            # the two lips, corner to corner
        upper, lower = (pA, pB) if B0[pA, 2].mean() > B0[pB, 2].mean() else (pB, pA)
        xc = max(abs(B0[L, 0].max()), abs(B0[L, 0].min()))
        def interp(path, x):
            px = B0[path, 0]; o = np.argsort(px)
            return np.array([np.interp(x, px[o], B0[path[o], ax]) for ax in range(3)])
        delta = np.zeros_like(B0)
        for path, other in ((upper, lower), (lower, upper)):
            for v in path:
                w = CC * float(np.clip((abs(B0[v, 0]) / xc - 0.45) / 0.45, 0, 1)) ** 2
                if w <= 0: continue
                q = interp(other, B0[v, 0]); dv = q - B0[v]; dn = np.linalg.norm(dv)
                if dn < 1e-9: continue
                tgt = max(0.0006, (1 - w) * dn)                    # leave a 0.6 mm seam so no face collapses
                delta[v] = 0.5 * max(0.0, dn - tgt) * dv / dn
        for kb in kbs:
            C = np.empty(len(B0) * 3, np.float32); kb.data.foreach_get('co', C)
            kb.data.foreach_set('co', (C.reshape(-1, 3) + delta).astype(np.float32).ravel())
        print('mouth corners: contour', len(L), 'verts, moved', int((np.abs(delta).sum(1) > 0).sum()), 'max mm', round(float(np.linalg.norm(delta, axis=1).max() * 1000), 2))
# ---- relax the mouth region of every shape: smoothing the displacement (not the surface) removes local folds
_E = np.array([e.vertices[:] for e in head.data.edges]); _nv = len(head.data.vertices)
import scipy.sparse as _sp2
_A = _sp2.csr_matrix((np.ones(2 * len(_E)), (np.r_[_E[:, 0], _E[:, 1]], np.r_[_E[:, 1], _E[:, 0]])), shape=(_nv, _nv))
_dg = np.asarray(_A.sum(1)).ravel(); _dg[_dg == 0] = 1; _AN = _sp2.diags(1 / _dg) @ _A
_B0 = np.array([c.co[:] for c in kbs[0].data]); _mc = conv(Vn[[LMK['mouth_corner']]])[0] * np.array([0, 1, 1])
_w = np.clip((0.055 - np.linalg.norm(_B0 - _mc, axis=1)) / 0.02, 0, 1)[:, None] * CFG.get('mouth_relax', 0.6)
_EC = np.array(list(head['eye_centers'])) if 'eye_centers' in head.keys() else None
_we = np.zeros_like(_w)
for _c in ((EYE_CENTERS['Left'], EYE_CENTERS['Right']) if 'EYE_CENTERS' in globals() else ()):
    _we = np.maximum(_we, np.clip((0.030 - np.linalg.norm(_B0 - _c, axis=1)) / 0.012, 0, 1)[:, None] * 0.5)
for kb in kbs[1:]:
    C = np.array([c.co[:] for c in kb.data]); d = C - _B0
    ww = np.maximum(_w, _we) if kb.name.startswith(('eye', 'cheekSquint')) else _w
    for _ in range(CFG.get('mouth_relax_iters', 6)): d = d + ww * ((_AN @ d) - d)
    kb.data.foreach_set('co', (_B0 + d).astype(np.float32).ravel())
print('mouth region relaxed in', len(kbs) - 1, 'shapes')
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
            C[v] = C[v] - np.array(nrm) * (sd + 0.0015); fixed += 1        # push straight in (snapping to the nearest point stacks vertices)
    kb.data.foreach_set('co', C.ravel())
head.data.vertices.foreach_set('co', np.array([c.co[:] for c in kbs[0].data], np.float32).ravel())
_kn = {kb.name: kb for kb in kbs}
_B = np.array([c.co[:] for c in kbs[0].data]); _J = np.array([c.co[:] for c in _kn['jawOpen'].data]) - _B
_Cl = np.array([c.co[:] for c in _kn['mouthClose'].data]) - _B; _combo = _B + _J + _Cl
_keep = []
for f in outer_f:
    q = _combo[f]; nrm = np.cross(q[1] - q[0], q[2] - q[0])
    if nrm @ (q.mean(0) - _combo.mean(0)) > 0: _keep.append(f)
_bvc = BVHTree.FromPolygons([Vector(c) for c in _combo], _keep); _nfix = 0
for v in lin_v:
    loc, nrm, _, dist = _bvc.find_nearest(Vector(_combo[v]))
    if loc is None or dist > 0.02: continue
    sd = (Vector(_combo[v]) - loc).dot(nrm)
    if sd > -0.0015: _Cl[v] -= np.array(nrm) * (sd + 0.0015); _nfix += 1
_kn['mouthClose'].data.foreach_set('co', (_B + _Cl).astype(np.float32).ravel())
print('lining tucked for jawOpen+mouthClose:', _nfix)
from scipy.spatial import cKDTree as _KDs
_mi = _KDs(_B * np.array([-1, 1, 1])).query(_B)[1]
def _mir(P_): return P_[_mi] * np.array([-1, 1, 1])
_Bs = 0.5 * (_B + _mir(_B)); _off = _Bs - _B
for kb in kbs:
    C = np.array([c.co[:] for c in kb.data])
    if kb.name.endswith('Right') and kb.name.replace('Right', 'Left') in _kn: continue
    C = C + _off
    if not kb.name.endswith('Left'): C = 0.5 * (C + _mir(C))
    kb.data.foreach_set('co', C.astype(np.float32).ravel())
for kb in kbs:
    if kb.name.endswith('Right') and kb.name.replace('Right', 'Left') in _kn:
        L_ = np.array([c.co[:] for c in _kn[kb.name.replace('Right', 'Left')].data])
        kb.data.foreach_set('co', _mir(L_).astype(np.float32).ravel())
head.data.vertices.foreach_set('co', np.array([c.co[:] for c in kbs[0].data], np.float32).ravel())
print('head made symmetric: neutral max fix mm', round(float(np.linalg.norm(_off, axis=1).max() * 1000), 2))

# ---- brows and lashes never go under the skin: per shape, push any vertex closer than the clearance back out
_polys = [list(p.vertices) for p in head.data.polygons]
_hk = {kb.name: np.array([c.co[:] for c in kb.data]) for kb in kbs}
for pk, clear, only in (('brows', 0.0005, None), ('lashes', 0.0006, None)):
    ob = PROXIES[pk]
    if not ob.data.shape_keys: continue
    moved = 0
    for pkb in ob.data.shape_keys.key_blocks:
        HC = _hk.get(pkb.name, _hk['Basis']); bv = BVHTree.FromPolygons([Vector(c) for c in HC], _polys)
        clear_k = clear
        Pp = np.array([c.co[:] for c in pkb.data])
        for i in range(len(Pp)):
            loc, nrm, _, dist = bv.find_nearest(Vector(Pp[i]))
            if loc is None or dist > 0.01: continue
            sd = (Vector(Pp[i]) - loc).dot(nrm)
            if sd < clear_k: Pp[i] = Pp[i] + np.array(nrm) * (clear_k - sd); moved += 1
        pkb.data.foreach_set('co', Pp.astype(np.float32).ravel())
        if pkb.name == 'Basis': ob.data.vertices.foreach_set('co', Pp.astype(np.float32).ravel())
    print(pk, 'skin clearance corrections:', moved)
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
