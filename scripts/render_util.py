import bpy, math
import numpy as np
from mathutils import Vector


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.threads_mode = 'AUTO'
    return sc


def mesh_from_arrays(name, V, F):
    """V (N,3) float, F (M,3|4) int."""
    me = bpy.data.meshes.new(name)
    V = np.asarray(V, np.float32); F = np.asarray(F, np.int32)
    k = F.shape[1]
    me.vertices.add(len(V)); me.loops.add(F.size); me.polygons.add(len(F))
    me.vertices.foreach_set('co', V.ravel())
    me.polygons.foreach_set('loop_start', np.arange(0, F.size, k, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(len(F), k, np.int32))
    me.loops.foreach_set('vertex_index', F.ravel())
    me.update(calc_edges=True); me.validate()
    me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def studio(sc, world=0.18):
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.9, 0.92, 1.0, 1); bg.inputs[1].default_value = world
    def light(name, loc, energy, size, col=(1, 1, 1)):
        l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size; l.color = col
        o = bpy.data.objects.new(name, l); sc.collection.objects.link(o)
        o.location = loc
        d = Vector((0, 0, 0.10)) - Vector(loc)
        o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return o
    light('key', (0.45, -0.75, 0.50), 16, 0.6, (1.0, 0.96, 0.9))
    light('fill', (-0.7, -0.45, 0.15), 4, 0.9, (0.85, 0.92, 1.0))
    light('rim', (-0.25, 0.75, 0.55), 12, 0.5)


def camera(sc, az_deg, elev_deg=0.0, target=(0, 0, 0.085), ortho=0.36, dist=1.5):
    cam = bpy.data.cameras.new('cam'); cam.type = 'ORTHO'; cam.ortho_scale = ortho
    o = bpy.data.objects.new('cam', cam); sc.collection.objects.link(o)
    az = math.radians(az_deg); el = math.radians(elev_deg)
    t = Vector(target)
    o.location = t + Vector((math.sin(az) * math.cos(el) * dist, -math.cos(az) * math.cos(el) * dist, math.sin(el) * dist))
    o.rotation_euler = (t - o.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = o
    return o


def render_views(sc, path_prefix, views, res=(640, 800), ortho=0.36, target=(0, 0, 0.085)):
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    outs = []
    for name, az, el in views:
        o = camera(sc, az, el, target=target, ortho=ortho)
        sc.render.filepath = f'{path_prefix}_{name}.png'
        bpy.ops.render.render(write_still=True)
        outs.append(sc.render.filepath)
        bpy.data.objects.remove(o)
    return outs


def sheet(paths, out, cols=None):
    from PIL import Image
    ims = [Image.open(p).convert('RGB') for p in paths]
    cols = cols or len(ims)
    rows = (len(ims) + cols - 1) // cols
    w, h = ims[0].size
    S = Image.new('RGB', (w * cols, h * rows), (30, 30, 30))
    for i, im in enumerate(ims):
        S.paste(im, ((i % cols) * w, (i // cols) * h))
    S.save(out)
    return out
