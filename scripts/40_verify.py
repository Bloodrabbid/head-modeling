import os
"""v2 checks on the shipped files: re-import every GLB and measure it; draw UV layout and wireframe sheets."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np, bpy, bmesh
from PIL import Image, ImageDraw
OUT = P.OUT + ''
rep = {}
for lod in ('head_lod0', 'head_lod1', 'head_lod2'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module='io_scene_gltf2')
    bpy.ops.import_scene.gltf(filepath=OUT + lod + '.glb')
    r = {}; total = 0
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.data.materials: continue
        me = o.data; bm = bmesh.new(); bm.from_mesh(me)
        t = sum(len(f.verts) - 2 for f in bm.faces); total += t
        mat = me.materials[0]
        d = {'tris': t, 'material': mat.name, 'double_sided': not mat.use_backface_culling,
             'textures': sorted({n.image.name + ' %dx%d' % tuple(n.image.size) for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image}),
             'shape_keys': len(me.shape_keys.key_blocks) - 1 if me.shape_keys else 0,
             'skinned_to': next((m.object.name for m in o.modifiers if m.type == 'ARMATURE' and m.object), None), 'vertex_groups': sorted(g.name for g in o.vertex_groups),
             'shape_values_on_import': sorted({round(k.value, 3) for k in me.shape_keys.key_blocks[1:]}) if me.shape_keys else []}
        if o.name.startswith('Head'):
            co = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
            d['size_m'] = [round(float(x), 3) for x in (co.max(0) - co.min(0))]
            d['zero_area_faces'] = int(sum(f.calc_area() < 1e-12 for f in bm.faces))
            d['non_manifold_edges'] = int(sum((not e.is_manifold) and (not e.is_boundary) for e in bm.edges))
            d['boundary_edges'] = int(sum(e.is_boundary for e in bm.edges))
            ul = bm.loops.layers.uv.active
            uv = np.array([[l[ul].uv[:] for l in f.loops] for f in bm.faces])
            e1 = uv[:, 1] - uv[:, 0]; e2 = uv[:, 2] - uv[:, 0]
            sa = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            d['uv_min_max'] = [round(float(uv.min()), 4), round(float(uv.max()), 4)]
            d['uv_area_sum'] = round(float(0.5 * np.abs(sa).sum()), 4)
            d['uv_flipped_tris'] = int((np.sign(sa) != np.sign(np.median(sa))).sum())
            key = set(map(tuple, np.round(co, 4)))
            d['asymmetric_verts'] = int(sum((round(-x, 4) + 0.0, y, z) not in key for x, y, z in map(tuple, np.round(co, 4))))
            if me.shape_keys: d['shape_names'] = [k.name for k in me.shape_keys.key_blocks[1:]]
        bm.free(); r[o.name] = d
    r['skeleton'] = [b.name for o in bpy.data.objects if o.type == 'ARMATURE' for b in o.data.bones]
    r['total_tris'] = total; r['file_mb'] = round(os.path.getsize(OUT + lod + '.glb') / 1e6, 2)
    rep[lod] = r
rep['baked_uv_coverage'] = round(json.load(open(P.WORK + 'stage2.json'))['coverage'], 4)
json.dump(rep, open(OUT + 'verification.json', 'w'), indent=1, ensure_ascii=False)
for lod in ('head_lod0', 'head_lod1', 'head_lod2'):
    h = [v for k, v in rep[lod].items() if k.startswith('Head')][0]
    print(lod, 'total', rep[lod]['total_tris'], {k: v['tris'] for k, v in rep[lod].items() if isinstance(v, dict)}, rep[lod]['file_mb'], 'MB')
    print('   head:', {k: h[k] for k in ('size_m', 'zero_area_faces', 'non_manifold_edges', 'boundary_edges', 'uv_area_sum', 'uv_min_max', 'uv_flipped_tris', 'asymmetric_verts', 'shape_keys', 'shape_values_on_import')})
print('coverage', rep['baked_uv_coverage'])

bpy.ops.wm.open_mainfile(filepath=OUT + 'head_source.blend')
me = bpy.data.objects['Head_source'].data
S = 1400
im = Image.open(OUT + 'textures/Head_BaseColor.png').resize((S, S)).convert('RGB'); dr = ImageDraw.Draw(im)
uv = me.uv_layers.active.data
for p in me.polygons:
    pts = [(uv[i].uv[0] * S, (1 - uv[i].uv[1]) * S) for i in p.loop_indices]
    dr.line(pts + [pts[0]], fill=(20, 40, 90), width=1)
im.save(OUT + 'preview_uv_layout.png')
# pole statistics of the quad source mesh
bm = bmesh.new(); bm.from_mesh(me)
val = {}
for v in bm.verts:
    if not v.is_boundary: val[len(v.link_edges)] = val.get(len(v.link_edges), 0) + 1
print('source head: faces', len(bm.faces), 'sides', {n: sum(1 for f in bm.faces if len(f.verts) == n) for n in (3, 4, 5)}, 'valence', dict(sorted(val.items())))
