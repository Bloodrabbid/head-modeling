import os
"""Preview renders of a .blend made by the v2 pipeline (perspective camera, like the reference viewer)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy, numpy as np
from mathutils import Vector
import render_util as R

def simple_materials():
    for ob in bpy.data.objects:
        if ob.type != 'MESH' or ob.data.materials: continue
        m = bpy.data.materials.new('pv_' + ob.name); m.use_nodes = True
        nt = m.node_tree; b = nt.nodes['Principled BSDF']
        tex = ob.get('texture')
        if tex and os.path.exists(tex):
            t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(tex, check_existing=True)
            nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
            if t.image.depth == 32 and ob.name != 'Head':
                nt.links.new(t.outputs['Alpha'], b.inputs['Alpha'])
        b.inputs['Roughness'].default_value = 0.5
        ob.data.materials.append(m)

def cam_persp(sc, az, el, target, dist, lens=70):
    c = bpy.data.cameras.new('c'); c.lens = lens
    o = bpy.data.objects.new('c', c); sc.collection.objects.link(o)
    a, e = math.radians(az), math.radians(el); t = Vector(target)
    o.location = t + Vector((math.sin(a) * math.cos(e) * dist, -math.cos(a) * math.cos(e) * dist, math.sin(e) * dist))
    o.rotation_euler = (t - o.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = o; return o

def shots(sc, prefix, views, target, dist=0.75, res=(640, 760), samples=32, lens=70):
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = samples; sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    out = []
    for name, az, el in views:
        o = cam_persp(sc, az, el, target, dist, lens)
        sc.render.filepath = f'{prefix}_{name}.png'; bpy.ops.render.render(write_still=True)
        out.append(sc.render.filepath); bpy.data.objects.remove(o)
    return out

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    hide = sys.argv[3].split(',') if len(sys.argv) > 3 else []
    bpy.ops.wm.open_mainfile(filepath=src)
    sc = bpy.context.scene
    simple_materials(); R.studio(sc, world=0.35)
    for o in bpy.data.objects:
        if o.type == 'MESH': o.hide_render = o.name in hide
    h = bpy.data.objects['Head']; zs = [v.co.z for v in h.data.vertices]
    tgt = (0, 0, max(zs) - 0.125)
    p = shots(sc, out, [('front', 0, 0), ('q', 40, 0), ('side', 90, 0)], tgt)
    R.sheet(p, out + '_sheet.png')
