import os
"""Fit MakeHuman morph weights + one perspective camera per reference view so that MediaPipe landmarks of our
head land where MediaPipe finds them on the reference screenshots.

  python3 fit.py init  weights_in.json cams_out.json             # generic cameras for the first render
  python3 fit.py solve weights_in.json corr.json cams_in.json weights_out.json cams_out.json [lambda]
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, cv2
from scipy.optimize import least_squares
import mhlib as M

REFS = {'front': (P.REF['front'], 0.5),
        'right': (P.REF['right'], 1.0),     # profile, nose to image right = character's right side
        'left': (P.REF['left'], 1.0)}      # three-quarter-ish, character's left side
REFJ = {os.path.abspath(k): v for k, v in json.load(open(P.FIT + 'ref.json')).items()}
REFS = {k: (os.path.abspath(p), s) for k, (p, s) in REFS.items()}
V0 = M.load_obj(M.DATA + '/3dobjs/base.obj')[0]

def expand(tw):
    out = {}
    for k, w in tw.items():
        if '/lr-' in k: out[k.replace('/lr-', '/l-')] = w; out[k.replace('/lr-', '/r-')] = w
        else: out[k] = w
    return out

# ---- variables: signed pairs (decr/incr etc.) and one-sided shapes
PAIRS = []   # (name, neg_target, pos_target, lr)
def P(d, base, neg, pos, lr=False): PAIRS.append((f'{d}/{base}', f'{d}/{base}{neg}', f'{d}/{base}{pos}', lr))
for b in ('head-back-scale-depth', 'head-fat', 'head-scale-depth', 'head-scale-horiz', 'head-scale-vert'): P('head', b, '-decr', '-incr')
for b in ('forehead-nubian', 'forehead-scale-vert', 'forehead-temple'): P('forehead', b, '-decr', '-incr')
P('forehead', 'forehead-trans', '-backward', '-forward')
P('eyebrows', 'eyebrows-angle', '-down', '-up'); P('eyebrows', 'eyebrows-trans', '-down', '-up'); P('eyebrows', 'eyebrows-trans', '-backward', '-forward')
for b in ('eye-bag', 'eye-bag-height', 'eye-height1', 'eye-height2', 'eye-height3', 'eye-scale'): P('eyes', 'lr-' + b, '-decr', '-incr', True)
for b in ('eye-corner1', 'eye-corner2', 'eye-eyefold-angle', 'eye-eyefold', 'eye-trans'): P('eyes', 'lr-' + b, '-down', '-up', True)
for b in ('eye-bag', 'eye-epicanthus', 'eye-push1', 'eye-push2', 'eye-trans'): P('eyes', 'lr-' + b, '-in', '-out', True)
P('eyes', 'lr-eye-eyefold', '-concave', '-convex', True)
for b in ('nose-flaring', 'nose-greek', 'nose-hump', 'nose-nostrils-width', 'nose-point-width', 'nose-scale-depth', 'nose-scale-horiz',
          'nose-scale-vert', 'nose-septumangle', 'nose-volume', 'nose-width1', 'nose-width2', 'nose-width3'): P('nose', b, '-decr', '-incr')
for b in ('nose-base', 'nose-nostrils-angle', 'nose-point', 'nose-trans'): P('nose', b, '-down', '-up')
P('nose', 'nose-compression', '-compress', '-uncompress'); P('nose', 'nose-curve', '-concave', '-convex'); P('nose', 'nose-trans', '-backward', '-forward')
for b in ('mouth-cupidsbow', 'mouth-cupidsbow-width', 'mouth-lowerlip-height', 'mouth-lowerlip-volume', 'mouth-lowerlip-width', 'mouth-philtrum-volume',
          'mouth-scale-depth', 'mouth-scale-horiz', 'mouth-scale-vert', 'mouth-upperlip-height', 'mouth-upperlip-volume', 'mouth-upperlip-width'): P('mouth', b, '-decr', '-incr')
for b in ('mouth-angles', 'mouth-lowerlip-ext', 'mouth-lowerlip-middle', 'mouth-trans', 'mouth-upperlip-ext', 'mouth-upperlip-middle'): P('mouth', b, '-down', '-up')
P('mouth', 'mouth-trans', '-backward', '-forward')
for b in ('chin-bones', 'chin-height', 'chin-jaw-drop', 'chin-prognathism', 'chin-prominent', 'chin-width'): P('chin', b, '-decr', '-incr')
for b in ('cheek-bones', 'cheek-inner', 'cheek-volume'): P('cheek', 'lr-' + b, '-decr', '-incr', True)
P('cheek', 'lr-cheek-trans', '-down', '-up', True)
SINGLE = ['head/head-diamond', 'head/head-invertedtriangular', 'head/head-oval', 'head/head-rectangular', 'head/head-round',
          'head/head-square', 'head/head-triangular', 'chin/chin-triangle']
VAR_TARGETS = set()
for _, n, p_, lr in PAIRS: VAR_TARGETS |= {n, p_}
VAR_TARGETS |= set(SINGLE)

def target_delta(name):
    """Full-mesh delta (N,3) of a config-style target name (lr- expanded and summed)."""
    D = np.zeros_like(V0)
    for k in expand({name: 1.0}):
        i, o = M.load_target(M.TGT + '/' + k + '.target')
        if len(i): D[i] += o.reshape(-1, 3)
    return D

def split_weights(tw):
    fixed = {k: v for k, v in tw.items() if k not in VAR_TARGETS}
    x = []
    for _, n, p_, _ in PAIRS: x.append(tw.get(p_, 0.0) - tw.get(n, 0.0))
    for s in SINGLE: x.append(tw.get(s, 0.0))
    return fixed, np.array(x, float)

def join_weights(fixed, x, eps=0.02):
    tw = dict(fixed)
    for (_, n, p_, _), v in zip(PAIRS, x[:len(PAIRS)]):
        if v > eps: tw[p_] = round(float(v), 3)
        elif v < -eps: tw[n] = round(float(-v), 3)
    for s, v in zip(SINGLE, x[len(PAIRS):]):
        if v > eps: tw[s] = round(float(v), 3)
    return tw

def cam_lookat(C, T, f, w, h):
    fw = (T - C) / np.linalg.norm(T - C); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); dn = np.cross(fw, rt)
    R = np.stack([rt, dn, fw]); return {'R': R.tolist(), 't': (-R @ C).tolist(), 'f': f, 'w': w, 'h': h}

MID = None
def side_labels():
    L = np.array(REFJ[REFS['front'][0]]['pts'])[:468, :2]
    ipd = np.linalg.norm(L[468 - 468 + 33] - L[263])
    dx = L[:, 0] - L[168, 0]
    lab = np.where(np.abs(dx) < 0.03 * ipd, 0, np.sign(dx))      # image left (-1) = character's right side
    return lab
HAIRLINE = {10, 338, 297, 332, 284, 251, 389, 356, 454, 127, 162, 21, 54, 103, 67, 109, 234, 93, 323, 132, 361}

if sys.argv[1] == 'init':
    tw = json.load(open(sys.argv[2]))['targets']
    Vf = M.apply_targets(V0, expand(tw))
    S1 = json.load(open(P.WORK + 'stage1.json')); lm = S1['landmarks']
    T = np.array([0.0, (Vf[lm['brow'], 1] + Vf[lm['chin'], 1]) / 2, Vf[lm['nose_tip'], 2] - 0.9])
    cams = []
    for name, az, (w, h) in (('front', 0, (1354, 966)), ('right', -72, (1280, 896)), ('left', 66, (1280, 896))):
        a = math.radians(az); d = 9.0
        Tv = T if az == 0 else T + np.array([0.0, -0.15, 0.45])
        C = Tv + d * np.array([math.sin(a), 0.0, math.cos(a)])
        c = cam_lookat(C, Tv, (2.6 if az == 0 else 2.8) * w, w, h); c['name'] = name; cams.append(c)
    json.dump(cams, open(sys.argv[3], 'w'), indent=1); sys.exit()

# ------------------------------------------------------------------ solve
tw = json.load(open(sys.argv[2]))['targets']
CORR = {}
for cf in sys.argv[3].split(','):
    for k, v in json.load(open(cf)).items():
        if v: CORR[k] = v
_unused = None; CAM0 = {c['name']: c for c in json.load(open(sys.argv[4]))}
LAM = float(sys.argv[7]) if len(sys.argv) > 7 else 0.03
fixed, x0 = split_weights(tw)
Vbase = M.apply_targets(V0, expand(fixed))
DPOS = [target_delta(p_) for _, n, p_, _ in PAIRS]; DNEG = [target_delta(n) for _, n, p_, _ in PAIRS]; DS = [target_delta(s) for s in SINGLE]
lab = side_labels()

views = []
for name, (path, sc) in REFS.items():
    cr = CORR.get(name)
    if not cr: continue
    ref = np.array(REFJ[path]['pts'])[:478, :2] * sc
    keep = []
    for i in list(range(468)) + [468, 473]:
        c = cr['corr'][i] if i < len(cr['corr']) else None
        if i >= 468:
            if c is None: continue
            if name != 'front': continue                       # side screenshots: the character's gaze is turned, pupils are not where a straight gaze puts them
            keep.append(i); continue
        if c is None or i in HAIRLINE: continue
        if name == 'right' and lab[i] > 0: continue
        if name == 'left' and lab[i] < 0: continue
        if name != 'front' and c['facing'] < 0.05: continue
        keep.append(i)
    keep = np.array(keep)
    nmax = max(len(cr['corr'][i]['v']) for i in keep)
    vids = np.array([cr['corr'][i]['v'] + [0] * (nmax - len(cr['corr'][i]['v'])) for i in keep])
    bws = np.array([cr['corr'][i]['b'] + [0.0] * (nmax - len(cr['corr'][i]['b'])) for i in keep])
    const = np.array([cr['corr'][i].get('c', [0, 0, 0]) for i in keep])
    def gather(D): return (D[vids] * bws[..., None]).sum(1)
    Lb = gather(Vbase) + const
    EYE = {33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466}
    LIPS = {61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191}
    LWE = float(os.environ.get('EYE_W', '3'))
    lw = np.array([4.0 if i >= 468 else (LWE if i in EYE else (2.0 if i in LIPS else 1.0)) for i in keep])
    dP = np.stack([gather(D) for D in DPOS]); dN = np.stack([gather(D) for D in DNEG]); dS = np.stack([gather(D) for D in DS])
    scale = np.linalg.norm(ref[168] - ref[152])
    views.append(dict(name=name, keep=keep, lw=lw, ref=ref[keep], Lb=Lb, dP=dP, dN=dN, dS=dS, scale=scale, cam=CAM0[name], w=CAM0[name]['w'], h=CAM0[name]['h'],
                      wt=1.0 if name == 'front' else 0.8))
    print(name, 'landmarks used', len(keep), 'scale px', round(scale, 1))

# ---- silhouettes: the face's front edge (side views) and cheek/jaw edges (front view) from skin masks
S1 = json.load(open(P.WORK + 'stage1.json'))
import bpy_free_mesh as BM
HU, HE = BM.head_vertices_edges()                       # MakeHuman ids of the head vertices, edges as index pairs into HU
HB = Vbase[HU]
HDP = np.stack([D[HU] for D in DPOS]); HDN = np.stack([D[HU] for D in DNEG]); HDS = np.stack([D[HU] for D in DS])
for v in views:
    m = np.load(P.FIT + f"mask_{v['name']}.npy")
    ref_all = np.array(REFJ[REFS[v['name']][0]]['pts'])[:468, :2] * REFS[v['name']][1]
    sc_ = v['scale']; rows = []
    if v['name'] == 'front':
        y0, y1 = ref_all[1, 1], ref_all[152, 1] - 0.10 * sc_
        for y in np.arange(y0, y1, 3.0):
            xs = np.where(m[int(y)])[0]
            if len(xs) < 10: continue
            cx = ref_all[1, 0]; L_ = xs[xs < cx]; R_ = xs[xs > cx]
            if len(L_): rows.append((y, -1, L_.min()))
            if len(R_): rows.append((y, 1, R_.max()))
    else:
        y0, y1 = ref_all[6, 1], ref_all[152, 1] + 0.06 * sc_
        sgn = 1 if v['name'] == 'right' else -1
        for y in np.arange(y0, y1, 3.0):
            xs = np.where(m[int(y)])[0]
            if len(xs) < 10: continue
            rows.append((y, sgn, xs.max() if sgn > 0 else xs.min()))
    v['rows'] = np.array(rows)
    print(v['name'], 'silhouette rows', len(rows))

def silhouette_x(P2, rows, front_x=None):
    """For each (y, side) row: extreme x of the projected mesh edges crossing that row."""
    a = P2[HE[:, 0]]; b = P2[HE[:, 1]]
    y = rows[:, 0][:, None]
    dy = b[:, 1] - a[:, 1]; dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
    t = (y - a[:, 1]) / dy
    ok = (t >= 0) & (t <= 1)
    x = a[:, 0] + t * (b[:, 0] - a[:, 0])
    side = rows[:, 1][:, None]
    val = np.where(ok, x * side, -np.inf)
    if front_x is not None:                                   # front view: each side only from its own half of the face
        val = np.where(ok & ((x - front_x) * side > 0), val, -np.inf)
    out = val.max(1) * rows[:, 1]
    return out

NP = len(PAIRS)
def head_pts(x):
    xp, xs = x[:NP], x[NP:]
    return HB + np.tensordot(np.clip(xp, 0, None), HDP, 1) + np.tensordot(np.clip(-xp, 0, None), HDN, 1) + np.tensordot(np.clip(xs, 0, None), HDS, 1)
SIL_W = float(os.environ.get('SIL_W', '0.7'))
def shape_pts(v, x):
    xp, xs = x[:NP], x[NP:]
    return v['Lb'] + np.tensordot(np.clip(xp, 0, None), v['dP'], 1) + np.tensordot(np.clip(-xp, 0, None), v['dN'], 1) + np.tensordot(np.clip(xs, 0, None), v['dS'], 1)

def project(L, cp, w, h):
    R, _ = cv2.Rodrigues(cp[:3]); Xc = L @ R.T + cp[3:6]; f = math.exp(cp[6])
    return np.stack([f * Xc[:, 0] / Xc[:, 2] + w / 2, f * Xc[:, 1] / Xc[:, 2] + h / 2], 1)

def cam_params(c):
    rv, _ = cv2.Rodrigues(np.array(c['R'])); return np.concatenate([rv.ravel(), c['t'], [math.log(c['f'])]])

# initial camera per view: PnP over a range of focal lengths on the current shape
CP = []
for v in views:
    L = shape_pts(v, x0); best = None
    for fm in (1.0, 1.5, 2.0, 2.6, 3.5, 5.0, 7.0):
        f = fm * v['w']; K = np.array([[f, 0, v['w'] / 2], [0, f, v['h'] / 2], [0, 0, 1]])
        ok, rv, tv = cv2.solvePnP(L, v['ref'], K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        cp = np.concatenate([rv.ravel(), tv.ravel(), [math.log(f)]])
        e = np.sqrt(((project(L, cp, v['w'], v['h']) - v['ref']) ** 2).sum(1)).mean() / v['scale']
        if best is None or e < best[0]: best = (e, cp)
    print(v['name'], 'PnP start error (face-height units)', round(best[0], 4), 'f/w', round(math.exp(best[1][6]) / v['w'], 2))
    CP.append(best[1])

def unpack(z): return z[:len(x0)], [z[len(x0) + 7 * k: len(x0) + 7 * k + 7] for k in range(len(views))]

def residuals(z):
    x, cps = unpack(z); r = []
    for v, cp in zip(views, cps):
        r.append(((project(shape_pts(v, x), cp, v['w'], v['h']) - v['ref']) / v['scale'] * v['wt'] * v['lw'][:, None]).ravel())
        if SIL_W > 0 and len(v['rows']):
            P2 = project(head_pts(x), cp, v['w'], v['h'])
            fx = project(shape_pts(v, x), cp, v['w'], v['h'])[:, 0].mean() if v['name'] == 'front' else None
            sx = silhouette_x(P2, v['rows'], fx)
            d = np.where(np.isfinite(sx), sx - v['rows'][:, 2], 0.0)
            r.append(np.clip(d / v['scale'], -0.3, 0.3) * SIL_W * math.sqrt(len(v['keep']) / max(1, len(v['rows']))) * 0.5)
        r.append([0.02 * (cp[6] - math.log(2.0 * v['w']))])                    # weak prior on focal length
    TIE = float(os.environ.get('TIE_F', '0'))
    if TIE > 0:
        lf = np.array([cp[6] - math.log(v['w']) for v, cp in zip(views, cps)])
        r.append(TIE * (lf - lf.mean()))
    r.append(LAM * x)
    return np.concatenate(r)

def report(z, tag):
    x, cps = unpack(z); out = {}
    for v, cp in zip(views, cps):
        e = np.sqrt(((project(shape_pts(v, x), cp, v['w'], v['h']) - v['ref']) ** 2).sum(1)) / v['scale']
        P2 = project(head_pts(x), cp, v['w'], v['h'])
        fx = project(shape_pts(v, x), cp, v['w'], v['h'])[:, 0].mean() if v['name'] == 'front' else None
        sx = silhouette_x(P2, v['rows'], fx); d = np.abs(sx - v['rows'][:, 2]) / v['scale']; d = d[np.isfinite(d)]
        out[v['name']] = (round(float(e.mean()), 4), round(float(np.median(d)), 4), 'iris', round(float(e[v['keep'] >= 468].mean()), 4) if (v['keep'] >= 468).any() else None)
    print(tag, '(landmark mean, silhouette median) in face-height units:', out); return out

BND = float(os.environ.get('BND', '1.0'))
z0 = np.concatenate([x0] + CP)
lo = np.concatenate([[-BND] * NP + [0.0] * len(SINGLE)] + [[-np.inf] * 6 + [math.log(0.6 * v['w'])] for v in views])
hi = np.concatenate([[BND] * NP + [1.0] * len(SINGLE)] + [[np.inf] * 6 + [math.log(12 * v['w'])] for v in views])
z0 = np.clip(z0, lo + 1e-9, hi - 1e-9)
e0 = report(z0, 'before')
# cameras alone first, then everything
mask = np.zeros(len(z0), bool); mask[len(x0):] = True
def res_cam(zc): z = z0.copy(); z[mask] = zc; return residuals(z)
rc = least_squares(res_cam, z0[mask], bounds=(lo[mask], hi[mask]), x_scale='jac', max_nfev=200)
z1 = z0.copy(); z1[mask] = rc.x
report(z1, 'cameras only')
r = least_squares(residuals, z1, bounds=(lo, hi), x_scale='jac', max_nfev=int(os.environ.get('NFEV', '80')), verbose=0)
e1 = report(r.x, 'after')
x, cps = unpack(r.x)
tw_new = join_weights(fixed, x)
json.dump({'targets': tw_new}, open(sys.argv[5], 'w'), indent=1)
cams = []
for v, cp in zip(views, cps):
    R, _ = cv2.Rodrigues(cp[:3]); cams.append({'name': v['name'], 'R': R.tolist(), 't': cp[3:6].tolist(), 'f': math.exp(cp[6]), 'w': v['w'], 'h': v['h']})
json.dump(cams, open(sys.argv[6], 'w'), indent=1)
ch = sorted([(abs(b - a), n, round(a, 2), round(b, 2)) for n, a, b in zip([p[0] for p in PAIRS] + SINGLE, x0, x)], reverse=True)[:25]
print('largest changes:'); [print(f'  {n}: {a} -> {b}') for _, n, a, b in ch]
for v, cp in zip(views, cps):
    C = -cv2.Rodrigues(cp[:3])[0].T @ cp[3:6]
    print(v['name'], 'f/w', round(math.exp(cp[6]) / v['w'], 2), 'camera distance dm', round(float(np.linalg.norm(C - v['Lb'].mean(0))), 2))
