import os
"""v2, stage 2: re-atlas the head's UV islands into a dedicated 2048 texture, bake position + AO,
compose BaseColor / ORM / Normal, and prepare the small textures of the proxies."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import mhlib as M

W = P.WORK + ''; OUT = P.OUT + 'textures/'
os.makedirs(OUT, exist_ok=True)
RES = 2048
CFG = json.load(open(P.CONFIG)); S1 = json.load(open(W + 'stage1.json'))
bpy.ops.wm.open_mainfile(filepath=W + 'stage1.blend')
sc = bpy.context.scene
head = bpy.data.objects['Head']; me = head.data
nl = len(me.loops)
uv = np.empty(nl * 2, np.float32); me.uv_layers.active.data.foreach_get('uv', uv); uv = uv.reshape(nl, 2).astype(np.float64)
lv = np.empty(nl, np.int32); me.loops.foreach_get('vertex_index', lv)

# ---- UV islands by union-find over (vertex, uv) corners
key = {}
corner = np.empty(nl, np.int64)
for i in range(nl):
    k = (int(lv[i]), round(uv[i, 0], 5), round(uv[i, 1], 5))
    corner[i] = key.setdefault(k, len(key))
par = list(range(len(key)))
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for p in me.polygons:
    ls = list(p.loop_indices); r = find(corner[ls[0]])
    for l in ls[1:]:
        q = find(corner[l])
        if q != r: par[q] = r
isl_of_loop = np.array([find(c) for c in corner])
ids = np.unique(isl_of_loop)
islands = []
for k in ids:
    m = isl_of_loop == k
    b = (uv[m, 0].min(), uv[m, 0].max(), uv[m, 1].min(), uv[m, 1].max())
    islands.append({'id': int(k), 'mask': m, 'bbox': b, 'area': (b[1] - b[0]) * (b[3] - b[2]), 'n': int(m.sum())})
islands.sort(key=lambda d: -d['area'])
# split the head island: the face (front of the ears) keeps most of the atlas, the back of the head and the ears
# (under the hair) become their own, smaller island
if CFG.get('uv_split_face', True):
    pol = np.empty(nl, np.int64)
    for p in me.polygons: pol[list(p.loop_indices)] = p.index
    pc = np.array([p.center[:] for p in me.polygons]); HVy = np.array([v.co[1] for v in me.vertices])
    y_split = HVy.min() + CFG.get('uv_face_depth', 0.105)          # metres behind the front-most point of the face
    _ez = np.array([v.co[2] for v in bpy.data.objects['Eyes'].data.vertices]).mean()
    hd0 = islands[0]; front = (pc[pol, 1] < y_split) & (pc[pol, 2] < _ez + CFG.get('uv_crown_above_eyes', 0.075))   # crown (under the hair) -> back
    parts = []
    for m in (hd0['mask'] & front, hd0['mask'] & ~front):
        b_ = (uv[m, 0].min(), uv[m, 0].max(), uv[m, 1].min(), uv[m, 1].max())
        parts.append({'id': hd0['id'], 'mask': m, 'bbox': b_, 'area': (b_[1] - b_[0]) * (b_[3] - b_[2]), 'n': int(m.sum())})
    islands = [parts[0], islands[1], parts[1]] + islands[2:]
    print('head island split: face loops', parts[0]['n'], 'back loops', parts[1]['n'])
INSIDE_FROM = 3 if CFG.get('uv_split_face', True) else 2
print('islands:', [(d['n'], [round(x, 3) for x in d['bbox']]) for d in islands])

# ---- layout: the face island upright across the top half, everything else on shelves below
MG = 0.008
FH = CFG.get('uv_face_fraction', 0.62) if CFG.get('uv_split_face', True) else 0.5
hd = islands[0]; u0, u1, v0, v1 = hd['bbox']; du, dv = u1 - u0, v1 - v0
s_h = min((1 - 2 * MG) / dv, (FH - 2 * MG) / du)
hd['T'] = ('rot', s_h, MG, (1 - FH) + MG + ((FH - 2 * MG) - du * s_h) / 2)
face_x1 = MG + dv * s_h                                          # right edge of the face island in the atlas
FREE = [(MG, MG, 1 - MG, 1 - FH - MG)]                              # bottom strip
if 1 - face_x1 > 0.08: FREE.append((face_x1 + MG, 1 - FH + MG, 1 - MG, 1 - MG))   # column right of the face
PRIO = {1: 3.2, 2: 0.8}                                             # neck strip is visible: denser; back of head under the hair
def place(scale):
    order = sorted(range(1, len(islands)), key=lambda i: -(islands[i]['bbox'][3] - islands[i]['bbox'][2]) * PRIO.get(i, 0.8))
    cur = [[r[0], r[1], 0.0] for r in FREE]; out = {}
    for i in order:
        a0, a1, b0, b1 = islands[i]['bbox']; sc_ = scale * PRIO.get(i, 0.8)
        w = (a1 - a0) * sc_; h = (b1 - b0) * sc_; ok = False
        for r, c in zip(FREE, cur):
            if c[0] + w > r[2]: c[0] = r[0]; c[1] += c[2] + MG; c[2] = 0.0
            if c[0] + w <= r[2] and c[1] + h <= r[3]:
                out[i] = ('flat', sc_, c[0], c[1]); c[0] += w + MG; c[2] = max(c[2], h); ok = True; break
        if not ok: return None
    return [out[i] for i in range(1, len(islands))]
s_o = s_h
while True:
    pl = place(s_o)
    if pl is not None: break
    s_o *= 0.96
for d, t in zip(islands[1:], pl): d['T'] = t
print('scale head %.2f others %.2f' % (s_h, s_o))

def fwd(d, U):                       # old uv -> atlas uv
    kind, s, ox, oy = d['T']; a0, a1, b0, b1 = d['bbox']
    if kind == 'rot': return np.stack([ox + (U[:, 1] - b0) * s, oy + (a1 - U[:, 0]) * s], -1)
    return np.stack([ox + (U[:, 0] - a0) * s, oy + (U[:, 1] - b0) * s], -1)
def inv(d, A):                       # atlas uv -> old uv
    kind, s, ox, oy = d['T']; a0, a1, b0, b1 = d['bbox']
    if kind == 'rot': return np.stack([a1 - (A[:, 1] - oy) / s, b0 + (A[:, 0] - ox) / s], -1)
    return np.stack([a0 + (A[:, 0] - ox) / s, b0 + (A[:, 1] - oy) / s], -1)

new = uv.copy()
for d in islands: new[d['mask']] = fwd(d, uv[d['mask']])
me.uv_layers.active.data.foreach_set('uv', new.astype(np.float32).ravel())
print('new uv range', new.min(0).round(3), new.max(0).round(3))

# ---- resample the MakeHuman skin into the atlas (array rows = v from bottom to top)
src = np.asarray(Image.open(head['texture']).convert('RGB'), np.float32)[::-1] / 255.0
SH, SW = src.shape[:2]
tint = np.array(CFG.get('skin_tint', [1.0, 0.95, 0.85]), np.float32)
atlas = np.tile((src.reshape(-1, 3).mean(0))[None, None], (RES, RES, 1)).astype(np.float32)
owner = np.full((RES, RES), -1, np.int32)
PAD = 10
for k, d in enumerate(islands):
    A = new[d['mask']]
    x0 = max(int(A[:, 0].min() * RES) - PAD, 0); x1 = min(int(A[:, 0].max() * RES) + PAD + 1, RES)
    y0 = max(int(A[:, 1].min() * RES) - PAD, 0); y1 = min(int(A[:, 1].max() * RES) + PAD + 1, RES)
    X, Y = np.meshgrid((np.arange(x0, x1) + 0.5) / RES, (np.arange(y0, y1) + 0.5) / RES)
    U = inv(d, np.stack([X.ravel(), Y.ravel()], -1))
    cx = U[:, 0] * SW - 0.5; cy = U[:, 1] * SH - 0.5
    for ch in range(3):
        atlas[y0:y1, x0:x1, ch] = ndi.map_coordinates(src[..., ch], [cy, cx], order=1, mode='nearest').reshape(y1 - y0, x1 - x0)
    owner[y0:y1, x0:x1] = k
atlas *= tint

# ---- bake position and AO on the new layout
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.use_denoising = False
for o in bpy.data.objects: o.hide_render = o.name not in ('Head', 'Teeth')
mat = bpy.data.materials.new('bake'); mat.use_nodes = True; head.data.materials.append(mat)
node = mat.node_tree.nodes.new('ShaderNodeTexImage'); mat.node_tree.nodes.active = node
def bake(kind, samples, margin):
    img = bpy.data.images.new('b_' + kind, RES, RES, alpha=True, float_buffer=True, is_data=True)
    node.image = img; sc.cycles.samples = samples
    bpy.ops.object.select_all(action='DESELECT'); head.select_set(True); bpy.context.view_layer.objects.active = head
    b = sc.render.bake; b.use_selected_to_active = False; b.margin = margin; b.margin_type = 'EXTEND'
    bpy.ops.object.bake(type=kind)
    a = np.empty(RES * RES * 4, np.float32); img.pixels.foreach_get(a)
    return a.reshape(RES, RES, 4)
if os.environ.get('REUSE_BAKE') and os.path.exists(W + 'bake_cache.npz'):        # same shape as last run: skip the slow bakes
    _c = np.load(W + 'bake_cache.npz'); pos, ao = _c['pos'], _c['ao']
else:
    pos = bake('POSITION', 1, 0)[..., :3]
    ao = bake('AO', 32, 16)[..., 0]
    np.savez_compressed(W + 'bake_cache.npz', pos=pos, ao=ao)
head.data.materials.clear()
cov = np.abs(pos).sum(-1) > 0
print('coverage %.1f%%' % (100 * cov.mean()))
P = pos[cov]

rng = np.random.default_rng(11)
LAT = rng.random((64, 64, 64)).astype(np.float32)
def vnoise(Q, scale, off=0.0):
    return ndi.map_coordinates(LAT, (((Q / scale) + off) % 64).T, order=1, mode='grid-wrap')
def sstep(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

# landmarks in Blender space
Vn = np.load(W + 'Vn.npy'); org = np.array(S1['origin_mh'])
def cv(Q): return M.to_blender(np.asarray(Q) - org)
LM = {k: cv(Vn[[v]])[0] for k, v in S1['landmarks'].items()}
def tverts(name, frac=0.3):
    i, o = M.load_target(M.TGT + '/' + name + '.target'); n = np.linalg.norm(o, axis=1)
    return cv(Vn[i[n > frac * n.max()]])
def vweight(name, sigma=0.0016):
    i, o = M.load_target(M.TGT + '/' + name + '.target'); n = np.linalg.norm(o, axis=1)
    tree = cKDTree(cv(Vn[i])); d, j = tree.query(P, k=8)
    g = np.exp(-(d / sigma) ** 2); val = (g * (n / n.max())[j]).sum(1) / np.maximum(g.sum(1), 1e-9)
    return np.where(d.min(1) < 0.004, val, 0.0)
up_w = vweight('mouth/mouth-upperlip-volume-incr'); lo_w = vweight('mouth/mouth-lowerlip-volume-incr')
LIP_T = CFG.get('lip_threshold', [0.22, 0.34]); LIP_TU = CFG.get('lip_threshold_upper', LIP_T)
up_m = sstep(LIP_TU[0], LIP_TU[1], up_w); lo_m = sstep(LIP_T[0], LIP_T[1], lo_w)
lipm = np.maximum(up_m, lo_m)
np.save(W + 'lipw.npy', np.stack([up_w, lo_w]))
A = P.copy(); A[:, 0] = np.abs(A[:, 0])
def gauss(c, s): d = A - np.asarray(c, np.float32); return np.exp(-(d * d).sum(-1) / (2 * s * s))
cheek = np.array([abs(LM['cheekbone'][0]) * 0.72, LM['cheekbone'][1] - 0.012, LM['alar'][2] + 0.012])
nose_b = np.array([0.0, LM['nose_tip'][1] + 0.012, LM['alar'][2] + 0.022])
forehead_c = np.array([abs(LM['cheekbone'][0]) * 0.35, LM['brow'][1] - 0.004, LM['brow'][2] + 0.035])
frk_zone = np.clip(1.0 * gauss(cheek, 0.020) + 0.8 * gauss(nose_b, 0.014) + 0.15 * gauss(forehead_c, 0.022), 0, 1)
inside = owner[cov] >= INSIDE_FROM                         # islands other than face, neck and back of head: mouth interior etc.

col = atlas[cov].copy()
lum = col.mean(1, keepdims=True); col = np.clip(lum + (col - lum) * 0.88, 0, 1)            # a little more saturation
col *= (1 + 0.035 * (vnoise(P, 0.006) - 0.5) * 2)[:, None]
col += (gauss(cheek, 0.026) * (~inside))[:, None] * np.array([0.030, -0.022, -0.018])        # blush
k = (0.80 * up_m * (~inside))[:, None]; col = col * (1 - k) + col * np.array(CFG.get('lip_upper', [0.70, 0.44, 0.47])) * k
k = (0.75 * lo_m * (1 - up_m) * (~inside))[:, None]; col = col * (1 - k) + col * np.array(CFG.get('lip_lower', [0.92, 0.64, 0.68])) * k
col_nofrk = col.copy()
f1 = sstep(0.77, 0.82, vnoise(P, 0.0006)) * sstep(0.30, 0.60, vnoise(P, 0.010, 5.1))        # freckles: dots gated by patches
f2 = sstep(0.88, 0.91, vnoise(P, 0.0010, 9.7))
frk = np.clip(f1 + 0.8 * f2, 0, 1) * frk_zone * (1 - lipm) * (~inside)
k = (0.95 * frk)[:, None]; col = col * (1 - k) + col * np.array([0.58, 0.40, 0.30]) * k
col[inside] *= 0.45                                       # mouth interior: darker, it sits in shadow
col = np.clip(col, 0.02, 0.97)

tz = np.clip(gauss((0, LM['brow'][1], LM['brow'][2] + 0.03), 0.03) + gauss(LM['nose_tip'], 0.016), 0, 1)
rough = 0.52 + 0.06 * (vnoise(P, 0.004, 2.2) - 0.5) - 0.09 * tz
rough = rough * (1 - lipm) + (0.55 * up_m + 0.45 * lo_m * (1 - up_m)) / np.maximum(lipm, 1e-6) * lipm
rough[inside] = 0.30
rough = np.clip(rough, 0.05, 1)

def fill(img):
    idx = ndi.distance_transform_edt(~cov, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]
base = atlas.copy(); base[cov] = col
base = fill(base)
# detail normal: albedo high-pass (wrinkles, lip lines) + pore noise, as a height field in texture space
smooth_src = atlas.copy(); smooth_src[cov] = np.clip(col_nofrk, 0, 1); g = fill(smooth_src).mean(-1)      # detail from skin without the freckles
hp = ndi.gaussian_filter(g, 1.2) - ndi.gaussian_filter(g, 6.0)
h = np.zeros((RES, RES), np.float32); h[cov] = 0.55 * vnoise(P, 0.00042) + 0.45 * vnoise(P, 0.0010, 3.3)
h = h * 0.9 + hp * 6.0
inner = ndi.binary_erosion(cov, iterations=2)
gy, gx = np.gradient(h); gx[~inner] = 0; gy[~inner] = 0
n = np.zeros((RES, RES, 3), np.float32); n[..., 2] = 1
STR = 1.1; n[..., 0] = -STR * gx; n[..., 1] = -STR * gy
n /= np.linalg.norm(n, axis=-1, keepdims=True)
flat = np.zeros((RES, RES), bool); flat[cov] = inside
nrm = n * 0.5 + 0.5; nrm[flat] = (0.5, 0.5, 1.0); nrm[~cov] = (0.5, 0.5, 1.0)
orm = np.zeros((RES, RES, 3), np.float32); orm[..., 0][cov] = np.clip(0.30 + 0.70 * ao[cov], 0, 1); orm[..., 1][cov] = rough
orm = fill(orm)

def save(name, arr, mode='RGB', size=None):
    im = Image.fromarray((np.clip(arr[::-1], 0, 1) * 255 + 0.5).astype(np.uint8), mode)
    if size: im = im.resize(size, Image.LANCZOS)
    im.save(OUT + name, optimize=True); print('saved', name, im.size)
save('Head_BaseColor.png', base); save('Head_ORM.png', orm); save('Head_Normal.png', nrm)

# ---- proxy textures: resized, tinted where the reference asks for it
def proxy_tex(ob, name, size, tint=(1, 1, 1), alpha_gain=1.0, rgb_only=False):
    a = np.asarray(Image.open(ob['texture']).convert('RGBA'), np.float32) / 255.0
    a[..., :3] *= np.array(tint, np.float32)
    a[..., 3] = np.clip(a[..., 3] * alpha_gain, 0, 1)
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA')
    if rgb_only: im = im.convert('RGB')
    im = im.resize((size, size), Image.LANCZOS); im.save(OUT + name, optimize=True); print('saved', name, im.size, im.mode)
O = bpy.data.objects
if O['Hair'].get('procedural'):                             # groomed hair: the cap gets dark streaks running down the head
    _r = np.random.default_rng(5); W_ = 512
    cols = ndi.gaussian_filter1d(_r.random(W_), 1.2); rows = np.linspace(0, 1, W_)[:, None]
    lum = 0.75 + 0.5 * (cols[None, :] - cols.mean()) / cols.std() * 0.25 + 0.05 * np.sin(rows * 60)
    capc = np.clip(np.array([0.060, 0.045, 0.038])[None, None, :] * lum[..., None], 0, 1)
    Image.fromarray((np.dstack([capc, np.ones((W_, W_, 1))]) * 255 + 0.5).astype(np.uint8), 'RGBA').save(OUT + 'Hair_BaseColor.png'); print('saved Hair_BaseColor.png (procedural cap)')
else:
    proxy_tex(O['Hair'], 'Hair_BaseColor.png', 1024, tint=CFG.get('hair_tint', (0.30, 0.25, 0.24)))
def clear_unused(name, ob, size):                                  # wipe texels no UV triangle uses (the pack's hair map carries a logo there)
    from PIL import ImageDraw, ImageFilter
    uv = np.empty(len(ob.data.loops) * 2, np.float32); ob.data.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    mask = Image.new('L', (size, size), 0); dr = ImageDraw.Draw(mask)
    for poly in ob.data.polygons:
        pts = [(float(uv[l][0]) * size, (1 - float(uv[l][1])) * size) for l in poly.loop_indices]
        dr.polygon(pts, fill=255, outline=255)
    mask = mask.filter(ImageFilter.MaxFilter(9)); im = Image.open(OUT + name).convert('RGBA'); a_ = np.array(im)
    m = np.array(mask) > 0; mean = a_[m][:, :3].mean(0).astype(np.uint8)
    a_[~m, :3] = mean; a_[~m, 3] = 0; Image.fromarray(a_, 'RGBA').save(OUT + name, optimize=True); print('cleared unused texels', name, round(float((~m).mean()), 3))
if not O['Hair'].get('procedural'): clear_unused('Hair_BaseColor.png', O['Hair'], 1024)
def strand_texture(name, W_=512, H_=1024, seed=3):
    """Four hair clumps side by side: a solid root half that breaks into wavy, tapering strands toward the tip."""
    rng_ = np.random.default_rng(seed); img = np.zeros((H_, W_, 4), np.float32)
    xs = np.arange(W_)[None, :]; ys = np.arange(H_)[:, None] / (H_ - 1)          # ys = 0 at the root (top of the image)
    base_c = np.array([0.055, 0.042, 0.035]); gw = W_ / 4
    for g in range(4):
        u = (xs - g * gw) / gw                                                    # 0..1 across this clump
        edge = np.clip(np.sin(np.clip(u, 0, 1) * np.pi) * 1.6, 0, 1)
        clump = g < 2                                                             # groups 0-1: scalp clumps; 2-3: loose strands only
        if clump:
            body = edge * (1 - np.clip((ys - 0.35) / 0.45, 0, 1)) * 0.92         # solid near the root, gone by 80 % of the length
            img[..., 3] = np.maximum(img[..., 3], np.where((u >= 0) & (u <= 1), body, 0))
        for k in range(70 if clump else 55):
            x0 = (g + rng_.uniform(0.06, 0.94)) * gw; amp = rng_.uniform(1, 5); fr = rng_.uniform(1.0, 3.0); ph = rng_.uniform(0, 6.3)
            th = rng_.uniform(1.4, 3.2) if clump else rng_.uniform(1.1, 2.2); end = rng_.uniform(0.6, 1.0)
            xc = x0 + amp * np.sin(ph + fr * 6.28 * ys) + rng_.uniform(-10, 10) * ys ** 2
            a_ = np.exp(-((xs - xc) / th) ** 2) * np.clip((end - ys) / 0.2, 0, 1) * edge   # nothing along the card edges
            img[..., 3] = 1 - (1 - img[..., 3]) * (1 - a_)
    img[..., 3] *= np.clip(ys / 0.05, 0, 1)                                      # soft start at the root
    shade = 1 + 0.35 * (rng_.random((1, W_)) - 0.5) + 0.15 * np.sin(ys * 40)       # per-column variation, faint banding along the length
    tip_c = np.array([0.13, 0.092, 0.068])                                         # dark brown tips, near-black roots
    col_ = base_c * (1 - ys[..., None] ** 1.3) + tip_c * ys[..., None] ** 1.3
    img[..., :3] = col_ * shade[..., None]
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA').save(OUT + name, optimize=True); print('saved', name)
if 'HairCards' in bpy.data.objects: strand_texture('HairCards_BaseColor.png')
proxy_tex(O['Eyebrows'], 'Eyebrows_BaseColor.png', 512, tint=(0.20, 0.16, 0.15), alpha_gain=1.3)
proxy_tex(O['Eyelashes'], 'Eyelashes_BaseColor.png', 512)
O['Eyes']['texture'] = M.PACK + '/eyes/materials/' + CFG.get('eye_texture', 'brownlight') + '_eye.png'
proxy_tex(O['Eyes'], 'Eyes_BaseColor.png', 512)
_ea = np.asarray(Image.open(OUT + 'Eyes_BaseColor.png').convert('RGBA')).astype(np.float32) / 255; _e = _ea[..., :3]
_mx = _e.max(-1); _sat = (_mx - _e.min(-1)) / np.maximum(_mx, 1e-6); _iris = np.clip((_sat - 0.15) / 0.15, 0, 1)[..., None]
_e = _e * (1 - _iris) + _e * np.array(CFG.get('iris_tint', [0.62, 0.55, 0.52])) * _iris
Image.fromarray((np.clip(np.dstack([_e, _ea[..., 3:]]), 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA').save(OUT + 'Eyes_BaseColor.png'); print('iris darkened (alpha kept)')
proxy_tex(O['Teeth'], 'Teeth_BaseColor.png', 512, rgb_only=True)
proxy_tex(O['Tongue'], 'Tongue_BaseColor.png', 256, rgb_only=True)
json.dump({'islands': [{'n': d['n'], 'bbox': d['bbox'], 'T': d['T']} for d in islands], 'coverage': float(cov.mean())}, open(W + 'stage2.json', 'w'), indent=1)
for o in bpy.data.objects: o.hide_render = False
bpy.ops.wm.save_as_mainfile(filepath=W + 'stage2.blend')
print('done')
