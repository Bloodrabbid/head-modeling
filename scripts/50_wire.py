import os
import bpy, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import view as Vw, render_util as R
bpy.ops.wm.open_mainfile(filepath=P.OUT + 'head_source.blend')
sc = bpy.context.scene
src = bpy.data.objects['Head_source']
R.studio(sc, world=0.4)
for o in list(sc.objects):
    if o.type == 'MESH' and o is not src: o.hide_render = True
src.hide_render = False
clay = bpy.data.materials.new('clay'); clay.use_nodes = True
b = clay.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (0.62, 0.62, 0.64, 1); b.inputs['Roughness'].default_value = 0.7
ink = bpy.data.materials.new('ink'); ink.use_nodes = True
b = ink.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (0.02, 0.02, 0.03, 1); b.inputs['Roughness'].default_value = 1
src.data.materials.clear(); src.data.materials.append(clay); src.data.materials.append(ink)
m = src.modifiers.new('w', 'WIREFRAME'); m.thickness = 0.0005; m.use_replace = False; m.material_offset = 1; m.use_even_offset = False
bb = [src.matrix_world @ v.co for v in src.data.vertices]; top = max(v.z for v in bb)
p = Vw.shots(sc, P.WORK + 'wire', [('front', 0, 0), ('q', 40, 0), ('side', 90, 0), ('back', 150, 5)], (0, -0.011, top - 0.14), dist=0.87, res=(640, 760), samples=24)
R.sheet(p, P.OUT + 'preview_wireframe.png')
