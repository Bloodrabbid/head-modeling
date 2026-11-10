import os
"""v2, stage 3: materials, LODs (with blend shapes carried over), export GLB / FBX / .blend, renders."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy, bmesh
from PIL import Image
from scipy.spatial import cKDTree
import render_util as R, view as Vw

W = P.WORK + ''; OUT = P.OUT + ''; TEX = OUT + 'textures/'
bpy.ops.wm.open_mainfile(filepath=W + 'stage2.blend')
CFG_ = json.load(open(P.CONFIG)); XS = CFG_.get('export_scale', 1.0)
if XS != 1.0:                       # real-world size: MakeHuman's default woman is small (IPD ~53 mm); scale about the neck-cut origin
    for o in bpy.data.objects:
        if o.type != 'MESH': continue
        if o.data.shape_keys:
            for kb in o.data.shape_keys.key_blocks:
                C_ = np.empty(len(kb.data) * 3, np.float32); kb.data.foreach_get('co', C_); kb.data.foreach_set('co', C_ * XS)
        C_ = np.empty(len(o.data.vertices) * 3, np.float32); o.data.vertices.foreach_get('co', C_); o.data.vertices.foreach_set('co', C_ * XS)
        o.data.update()
    print('export scale', XS)
sc = bpy.context.scene
for mod in ('io_scene_gltf2', 'io_scene_fbx'):
    try: bpy.ops.preferences.addon_enable(module=mod)
    except Exception as ex: print('addon', mod, ex)
O = bpy.data.objects
for ob in O:
    for k in ('texture', 'mhclo', 'orig_index'):
        if k in ob: del ob[k]

# ---------------------------------------------------------------- materials
def img(name, srgb=True):
    im = bpy.data.images.load(TEX + name, check_existing=True); im.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
    return im
def gltf_ao_group():
    g = bpy.data.node_groups.get('glTF Material Output')
    if g is None:
        g = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
        g.interface.new_socket(name='Occlusion', in_out='INPUT', socket_type='NodeSocketFloat'); g.nodes.new('NodeGroupInput')
    return g
def material(name, base, alpha=None, orm=None, normal=None, rough=0.5, cull=True, spec=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = img(base)
    nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = rough
    if spec is not None: b.inputs['Specular IOR Level'].default_value = spec
    if alpha == 'BLEND':
        nt.links.new(t.outputs['Alpha'], b.inputs['Alpha'])
    elif alpha == 'MASK':                                       # Alpha -> Round -> Alpha is exported as alphaMode MASK, cutoff 0.5
        r = nt.nodes.new('ShaderNodeMath'); r.operation = 'ROUND'
        nt.links.new(t.outputs['Alpha'], r.inputs[0]); nt.links.new(r.outputs[0], b.inputs['Alpha'])
    if orm:
        to = nt.nodes.new('ShaderNodeTexImage'); to.image = img(orm, False)
        sep = nt.nodes.new('ShaderNodeSeparateColor'); grp = nt.nodes.new('ShaderNodeGroup'); grp.node_tree = gltf_ao_group()
        nt.links.new(to.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], b.inputs['Roughness']); nt.links.new(sep.outputs['Blue'], b.inputs['Metallic'])
        nt.links.new(sep.outputs['Red'], grp.inputs['Occlusion'])
    if normal:
        tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img(normal, False); nm = nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(tn.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    m.use_backface_culling = cull
    return m
MATS = {
    'Head': material('M_Head', 'Head_BaseColor.png', orm='Head_ORM.png', normal='Head_Normal.png', cull=False),
    'Eyes': material('M_Eyes', 'Eyes_BaseColor.png', rough=0.25),
    'Teeth': material('M_Teeth', 'Teeth_BaseColor.png', rough=0.25),
    'Tongue': material('M_Tongue', 'Tongue_BaseColor.png', rough=0.3),
    'Eyelashes': material('M_Eyelashes', 'Eyelashes_BaseColor.png', alpha='BLEND', rough=0.6, cull=False),
    'Eyebrows': material('M_Eyebrows', 'Eyebrows_BaseColor.png', alpha='BLEND', rough=0.7, cull=False),
    'Hair': material('M_Hair', 'Hair_BaseColor.png', alpha='MASK', rough=0.6, cull=False, spec=0.25),
}
if 'HairCards' in O: MATS['HairCards'] = material('M_HairCards', 'HairCards_BaseColor.png', alpha='MASK', rough=0.62, cull=False, spec=0.25)
MATS['Eyes_low'] = MATS['Eyes']
for n, m in MATS.items():
    O[n].data.materials.clear(); O[n].data.materials.append(m)

# ---- eyes: drop the transparent cornea shell so the eyeball is one opaque layer (no alpha sorting in engines)
def drop_transparent(ob, tex):
    a = np.asarray(Image.open(TEX + tex).convert('RGBA'))[::-1, :, 3]
    bm = bmesh.new(); bm.from_mesh(ob.data); ul = bm.loops.layers.uv.active
    kill = []
    for f in bm.faces:
        u = sum(l[ul].uv.x for l in f.loops) / len(f.loops); v = sum(l[ul].uv.y for l in f.loops) / len(f.loops)
        if a[min(int(v * a.shape[0]), a.shape[0] - 1), min(int(u * a.shape[1]), a.shape[1] - 1)] < 128: kill.append(f)
    n = len(kill); bmesh.ops.delete(bm, geom=kill, context='FACES'); bm.to_mesh(ob.data); bm.free()
    print(ob.name, 'transparent faces removed:', n)
drop_transparent(O['Eyes'], 'Eyes_BaseColor.png'); drop_transparent(O['Eyes_low'], 'Eyes_BaseColor.png')

# ---------------------------------------------------------------- LOD builders
def basis(ob):
    me = ob.data; n = len(me.vertices); a = np.empty(n * 3, np.float32)
    (me.shape_keys.key_blocks[0].data if me.shape_keys else me.vertices).foreach_get('co', a)
    return a.reshape(n, 3)
def tri_copy(src, name):
    """Triangulated copy that keeps UVs and shape keys (bmesh carries the shape layers)."""
    o = src.copy(); o.data = src.data.copy(); o.name = name; o.data.name = name; sc.collection.objects.link(o)
    bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.triangulate(bm, faces=bm.faces[:]); bm.to_mesh(o.data); bm.free()
    return o
def half_cards(src, name):
    """LOD copy of the hair cards that keeps every other card (cards are separate pieces)."""
    o = tri_copy(src, name); bm = bmesh.new(); bm.from_mesh(o.data); bm.verts.ensure_lookup_table()
    seen = set(); comps = []
    for v in bm.verts:
        if v.index in seen: continue
        stack = [v]; comp = []
        while stack:
            q = stack.pop()
            if q.index in seen: continue
            seen.add(q.index); comp.append(q)
            stack.extend(e.other_vert(q) for e in q.link_edges)
        comps.append(comp)
    drop = [v for k, c in enumerate(comps) if k % 2 for v in c]
    bmesh.ops.delete(bm, geom=drop, context='VERTS'); bm.to_mesh(o.data); bm.free()
    return o
def quarter_cards(src, name):
    o = half_cards(src, name + '_tmp'); q = half_cards(o, name); bpy.data.objects.remove(o); return q
def decimated(src, name, ratio, keep_shapes=True, rigid=False):
    """Collapse-decimated copy; blend shapes are re-attached by copying each delta from the nearest source vertex."""
    o = src.copy(); o.data = src.data.copy(); o.name = name; sc.collection.objects.link(o)
    keys = None
    if src.data.shape_keys:
        B = basis(src); keys = []
        for kb in src.data.shape_keys.key_blocks[1:]:
            a = np.empty(len(B) * 3, np.float32); kb.data.foreach_get('co', a); keys.append((kb.name, a.reshape(-1, 3) - B))
        o.shape_key_clear()
    md = o.modifiers.new('dec', 'DECIMATE'); md.ratio = ratio; md.use_symmetry = True; md.symmetry_axis = 'X'; md.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    o.modifiers.clear(); o.data = me; me.name = name
    bm_ = bmesh.new(); bm_.from_mesh(me); bmesh.ops.dissolve_degenerate(bm_, dist=1e-6, edges=bm_.edges[:]); bm_.to_mesh(me); bm_.free()
    if keys and keep_shapes:
        P = basis(o); nn = cKDTree(B).query(P)[1]
        o.shape_key_add(name='Basis')
        comp = None
        if rigid:                                               # solid parts (teeth): every connected piece moves as one rigid body or not at all
            n = len(P); par = list(range(n))
            def find(i):
                while par[i] != i: par[i] = par[par[i]]; i = par[i]
                return i
            for e in o.data.edges:
                a_, b_ = find(e.vertices[0]), find(e.vertices[1])
                if a_ != b_: par[a_] = b_
            comp = np.array([find(i) for i in range(n)])
        for kn, d in keys:
            D = d[nn].astype(np.float64)
            if comp is not None:
                mv_src = np.linalg.norm(d, axis=1) > 1e-4
                if mv_src.any():
                    A = B[mv_src].astype(np.float64); Bk = A + d[mv_src]
                    ca, cb = A.mean(0), Bk.mean(0); U, S, Vt = np.linalg.svd((A - ca).T @ (Bk - cb))
                    Rm = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T; tr = cb - Rm @ ca
                    for c in np.unique(comp):
                        m = comp == c
                        D[m] = (P[m] @ Rm.T + tr - P[m]) if mv_src[nn[m]].mean() > 0.5 else 0.0
            o.shape_key_add(name=kn).data.foreach_set('co', (P + D).astype(np.float32).ravel())
        for kb in o.data.shape_keys.key_blocks[1:]: kb.value = 0.0
    return o
LOD = {0: [tri_copy(O['Head'], 'Head_LOD0'), tri_copy(O['Eyes'], 'Eyes_LOD0'), decimated(O['Teeth'], 'Teeth_LOD0', 0.35, rigid=True),
           tri_copy(O['Tongue'], 'Tongue_LOD0'), tri_copy(O['Eyelashes'], 'Eyelashes_LOD0'), tri_copy(O['Eyebrows'], 'Eyebrows_LOD0'),
           tri_copy(O['Hair'], 'Hair_LOD0')] + ([tri_copy(O['HairCards'], 'HairCards_LOD0')] if 'HairCards' in O else []),
       1: [decimated(O['Head'], 'Head_LOD1', 0.5), tri_copy(O['Eyes_low'], 'Eyes_LOD1'), decimated(O['Teeth'], 'Teeth_LOD1', 0.12, rigid=True),
           tri_copy(O['Tongue'], 'Tongue_LOD1'), tri_copy(O['Eyelashes'], 'Eyelashes_LOD1'), tri_copy(O['Eyebrows'], 'Eyebrows_LOD1'), decimated(O['Hair'], 'Hair_LOD1', 0.5)] + ([half_cards(O['HairCards'], 'HairCards_LOD1')] if 'HairCards' in O else []),
       2: [decimated(O['Head'], 'Head_LOD2', 0.2, keep_shapes=False), decimated(O['Eyes_low'], 'Eyes_LOD2', 1.0, keep_shapes=False),
           decimated(O['Eyebrows'], 'Eyebrows_LOD2', 1.0, keep_shapes=False), decimated(O['Hair'], 'Hair_LOD2', 0.25)] + ([quarter_cards(O['HairCards'], 'HairCards_LOD2')] if 'HairCards' in O and O['Hair'].get('procedural') else [])}
def tris(o): return sum(len(p.vertices) - 2 for p in o.data.polygons)
for k, obs in LOD.items():
    print('LOD%d' % k, {o.name: tris(o) for o in obs}, 'total', sum(tris(o) for o in obs))
# ---------------------------------------------------------------- skeleton: root > neck > head > eye_L / eye_R
S1j = json.load(open(W + 'stage1.json')); Vn_ = np.load(W + 'Vn.npy'); org_ = np.array(S1j['origin_mh'])
def cvp(Q): return (np.asarray(Q) - org_) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T * 0.1 * XS
HV = np.array([v.co[:] for v in LOD[0][0].data.vertices]); top_z = HV[:, 2].max()
chin = cvp(Vn_[S1j['landmarks']['chin']]); chin_u = np.array([0.0, chin[1] + 0.020 * XS, chin[2] - 0.014 * XS])
backn = HV[(HV[:, 2] > 0.06 * XS) & (HV[:, 2] < 0.085 * XS)]; nape = np.array([0.0, backn[:, 1].max(), 0.072 * XS])
dd_ = nape - chin_u; PN = np.array([0.0, -dd_[2], dd_[1]]); PN /= np.linalg.norm(PN)
if PN[2] < 0: PN = -PN
EV = np.array([v.co[:] for v in LOD[0][1].data.vertices])
_ec = np.array(list(O['Head_source']['eye_centers'] if 'Head_source' in O else O['Head']['eye_centers'])) * XS
EYE_C = {'L': _ec[:3], 'R': _ec[3:]}
neck_y = HV[np.abs(HV[:, 2]) < 0.01 * XS][:, 1].mean()
arm_d = bpy.data.armatures.new('Head_Rig'); ARM = bpy.data.objects.new('Head_Rig', arm_d); sc.collection.objects.link(ARM)
bpy.ops.object.select_all(action='DESELECT'); bpy.context.view_layer.objects.active = ARM; ARM.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
def bone(name, h, t, parent=None):
    b = arm_d.edit_bones.new(name); b.head = tuple(h); b.tail = tuple(t)
    if parent: b.parent = arm_d.edit_bones[parent]
    return b
bone('root', (0, neck_y, 0), (0, neck_y, 0.03 * XS))
PIV = np.array([0.0, (EYE_C['L'][1] + 0.084 * XS), 0.10 * XS])               # head pivot near the skull base, behind the ear line
bone('neck', (0, neck_y, 0.0), tuple(PIV), 'root')
bone('head', tuple(PIV), (0, PIV[1], top_z), 'neck')
for s_ in ('L', 'R'):
    c = EYE_C[s_]; bone('eye_' + s_, c, c + np.array([0, -0.025 * XS, 0]), 'head')
bpy.ops.object.mode_set(mode='OBJECT')
def sst(a, b, v): t = np.clip((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def skin(o):
    V = np.array([v.co[:] for v in o.data.vertices]); groups = {}
    if o.name.startswith('Head_'):
        wh = sst(-0.035 * XS, 0.0, (V - chin_u) @ PN); groups = {'head': wh, 'neck': 1 - wh}
    elif o.name.startswith('Eyes_'):
        groups = {'eye_L': (V[:, 0] > 0).astype(float), 'eye_R': (V[:, 0] <= 0).astype(float)}
    else:
        groups = {'head': np.ones(len(V))}
    for gname, w in groups.items():
        g = o.vertex_groups.get(gname) or o.vertex_groups.new(name=gname)
        for i in np.where(w > 1e-4)[0]: g.add([int(i)], float(w[i]), 'REPLACE')
    mw = o.matrix_world.copy(); o.parent = ARM; o.matrix_world = mw
    md = o.modifiers.new('Armature', 'ARMATURE'); md.object = ARM
for k_, obs in LOD.items():
    for o in obs: skin(o)
print('rig: bones', [b.name for b in arm_d.bones], 'neck/head split plane through chin and nape')
SRC = [O[n] for n in ('Head', 'Eyes', 'Eyes_low', 'Teeth', 'Tongue', 'Eyelashes', 'Eyebrows', 'Hair', 'HairCards') if n in O]
for o in SRC: o.name = o.name + '_source'; o.data.name = o.name

# ---------------------------------------------------------------- save / export
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=OUT + 'head_source.blend')
def export(objs, path, fmt):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if fmt == 'glb':
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_yup=True,
                                  export_morph=True, export_morph_normal=True, export_apply=False, export_tangents=True)
    else:
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={'MESH', 'ARMATURE'}, mesh_smooth_type='FACE',
                                 path_mode='COPY', embed_textures=True, add_leaf_bones=False)
    print('exported', os.path.basename(path), round(os.path.getsize(path) / 1e6, 2), 'MB')
for k, obs in LOD.items(): export(obs + [ARM], OUT + 'head_lod%d.glb' % k, 'glb')
try: export([o for k in LOD for o in LOD[k]] + [ARM], OUT + 'head_all_lods.fbx', 'fbx')
except Exception as ex: print('FBX export failed:', ex)

# ---------------------------------------------------------------- renders
R.studio(sc, world=0.4)
def show(objs):
    for o in bpy.data.objects:
        if o.type == 'MESH': o.hide_render = o not in objs
def pose(objs, **kw):
    for o in objs:
        if o.data.shape_keys:
            for kb in o.data.shape_keys.key_blocks[1:]: kb.value = kw.get(kb.name, 0.0)
H = LOD[0][0]; top = max(v.co.z for v in H.data.vertices)
face = (0, -0.02 * XS, top - 0.125 * XS)
show(LOD[0]); pose(LOD[0])
p = Vw.shots(sc, W + 'f_lod0', [('front', 0, 0), ('q', 40, 0), ('side', 90, 0), ('back', 150, 5)], face, dist=0.78 * XS, res=(640, 760), samples=48)
R.sheet(p, OUT + 'preview_lod0.png')
p = Vw.shots(sc, W + 'f_ref', [('front', 0, -2), ('right', -78, 2), ('left', 62, 0)], (0, -0.05 * XS, top - 0.15 * XS), dist=0.90 * XS, res=(760, 620), samples=64, lens=150)
R.sheet(p, W + 'ours_close.png')
ex = []
POSES = (('jaw', dict(jawOpen=1.0)), ('blink', dict(eyeBlinkLeft=1.0, eyeBlinkRight=1.0)),
         ('smile', dict(mouthSmileLeft=1.0, mouthSmileRight=1.0, cheekSquintLeft=0.4, cheekSquintRight=0.4)),
         ('surprise', dict(browInnerUp=1.0, browOuterUpLeft=0.7, browOuterUpRight=0.7, eyeWideLeft=1.0, eyeWideRight=1.0, jawOpen=0.35)),
         ('frown', dict(browDownLeft=1.0, browDownRight=1.0, mouthFrownLeft=0.8, mouthFrownRight=0.8, noseSneerLeft=0.5, noseSneerRight=0.5)),
         ('look', dict(eyeLookOutLeft=1.0, eyeLookInRight=1.0, eyeLookUpLeft=0.4, eyeLookUpRight=0.4)),
         ('pucker', dict(mouthPucker=1.0)), ('puff', dict(cheekPuff=1.0, mouthPressLeft=0.5, mouthPressRight=0.5)),
         ('jaw_side', dict(jawLeft=1.0, jawOpen=0.3, mouthLeft=0.5)), ('tongue', dict(jawOpen=0.6, tongueOut=1.0)))
for nm, kw in POSES:
    pose(LOD[0], **kw); ex += Vw.shots(sc, W + 'f_' + nm, [('q', 22, 0)], (0, -0.04 * XS, top - 0.14 * XS), dist=0.52 * XS, res=(440, 500), samples=32)
pose(LOD[0]); R.sheet(ex, OUT + 'preview_expressions.png')
pl = []
for k in (0, 1, 2):
    show(LOD[k]); pose(LOD[k]); pl += Vw.shots(sc, W + 'f_l%d' % k, [('q', 35, 0)], face, dist=0.78 * XS, res=(560, 680), samples=40)
R.sheet(pl, OUT + 'preview_lods.png')
print('done')
