"""v2, stage 2: re-atlas the head's UV islands into a dedicated 2048 texture, bake position + AO,
compose BaseColor / ORM / Normal, and prepare the small textures of the proxies."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bpy
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import mhlib as M

import paths as P
W = P.WORK; OUT = P.OUT + 'textures/'
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
print('islands:', [(d['n'], [round(x, 3) for x in d['bbox']]) for d in islands])

# ---- layout: the face island upright across the top half, everything else on shelves below
MG = 0.008
hd = islands[0]; u0, u1, v0, v1 = hd['bbox']; du, dv = u1 - u0, v1 - v0
s_h = min((1 - 2 * MG) / dv, (0.5 - 2 * MG) / du)
hd['T'] = ('rot', s_h, MG + ((1 - 2 * MG) - dv * s_h) / 2, 0.5 + MG + ((0.5 - 2 * MG) - du * s_h) / 2)
def place(scale):
    x = MG; y = MG; rowh = 0; out = []
    for d in islands[1:]:
        a0, a1, b0, b1 = d['bbox']; w = (a1 - a0) * scale; h = (b1 - b0) * scale
        if x + w + MG > 1: x = MG; y += rowh + MG; rowh = 0
        if y + h + MG > 0.5: return None
        out.append(('flat', scale, x, y)); x += w + MG; rowh = max(rowh, h)
    return out
s_o = s_h
while True:
    pl = place(s_o)
    if pl is not None: break
    s_o *= 0.92
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
pos = bake('POSITION', 1, 0)[..., :3]
ao = bake('AO', 32, 16)[..., 0]
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
lips = np.concatenate([tverts('mouth/mouth-upperlip-volume-incr'), tverts('mouth/mouth-lowerlip-volume-incr')])
dlip = cKDTree(lips).query(P)[0]
lipm = 1 - sstep(0.0015, 0.0045, dlip)
A = P.copy(); A[:, 0] = np.abs(A[:, 0])
def gauss(c, s): d = A - np.asarray(c, np.float32); return np.exp(-(d * d).sum(-1) / (2 * s * s))
cheek = np.array([abs(LM['cheekbone'][0]) * 0.72, LM['cheekbone'][1] - 0.012, LM['alar'][2] + 0.012])
nose_b = np.array([0.0, LM['nose_tip'][1] + 0.012, LM['alar'][2] + 0.022])
frk_zone = np.clip(gauss(cheek, 0.024) + 0.8 * gauss(nose_b, 0.016), 0, 1)
inside = owner[cov] >= 2                                   # islands other than face and neck: mouth interior etc.

col = atlas[cov].copy()
lum = col.mean(1, keepdims=True); col = np.clip(lum + (col - lum) * 1.0, 0, 1)            # a little more saturation
col *= (1 + 0.035 * (vnoise(P, 0.006) - 0.5) * 2)[:, None]
col += (gauss(cheek, 0.026) * (~inside))[:, None] * np.array([0.030, -0.022, -0.018])        # blush
k = (0.70 * lipm * (~inside))[:, None]; col = col * (1 - k) + col * np.array([0.86, 0.58, 0.64]) * k
col_nofrk = col.copy()
f1 = sstep(0.80, 0.84, vnoise(P, 0.0007)) * sstep(0.40, 0.70, vnoise(P, 0.010, 5.1))        # freckles: dots gated by patches
f2 = sstep(0.87, 0.90, vnoise(P, 0.0011, 9.7))
frk = np.clip(f1 + 0.8 * f2, 0, 1) * frk_zone * (1 - lipm) * (~inside)
k = (0.78 * frk)[:, None]; col = col * (1 - k) + col * np.array([0.66, 0.50, 0.42]) * k
col = np.clip(col, 0.02, 0.97)

tz = np.clip(gauss((0, LM['brow'][1], LM['brow'][2] + 0.03), 0.03) + gauss(LM['nose_tip'], 0.016), 0, 1)
rough = 0.52 + 0.06 * (vnoise(P, 0.004, 2.2) - 0.5) - 0.09 * tz
rough = rough * (1 - lipm) + 0.42 * lipm
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
STR = 0.8; n[..., 0] = -STR * gx; n[..., 1] = -STR * gy
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
clear_unused('Hair_BaseColor.png', O['Hair'], 1024)
proxy_tex(O['Eyebrows'], 'Eyebrows_BaseColor.png', 512, tint=(0.20, 0.16, 0.15), alpha_gain=1.6)
proxy_tex(O['Eyelashes'], 'Eyelashes_BaseColor.png', 512)
proxy_tex(O['Eyes'], 'Eyes_BaseColor.png', 512, tint=(0.80, 0.72, 0.70))
proxy_tex(O['Teeth'], 'Teeth_BaseColor.png', 512, rgb_only=True)
proxy_tex(O['Tongue'], 'Tongue_BaseColor.png', 256, rgb_only=True)
json.dump({'islands': [{'n': d['n'], 'bbox': d['bbox'], 'T': d['T']} for d in islands], 'coverage': float(cov.mean())}, open(W + 'stage2.json', 'w'), indent=1)
for o in bpy.data.objects: o.hide_render = False
bpy.ops.wm.save_as_mainfile(filepath=W + 'stage2.blend')
print('done')
