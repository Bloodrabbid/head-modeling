import os
"""Folded faces per blend shape: faces of Head_LOD0 whose normal turns more than 90 degrees from the rest pose
(and faces that collapse), for every shape at 1.0 and for the main lip-sync combinations. Writes out/folds.json."""
import json, numpy as np, bpy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
bpy.ops.wm.open_mainfile(filepath=P.OUT + 'head_source.blend')
H = bpy.data.objects['Head_LOD0']; kb = H.data.shape_keys.key_blocks
F = np.array([list(p.vertices) for p in H.data.polygons])
B = np.array([v.co[:] for v in kb[0].data])
D = {k.name: np.array([v.co[:] for v in k.data]) - B for k in kb[1:]}
def normals(V):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(n, axis=1); return n / np.maximum(a[:, None], 1e-20), a
N0, A0 = normals(B)
POSES = {k: {k: 1.0} for k in D}
POSES.update({'jawOpen+mouthClose': {'jawOpen': 1, 'mouthClose': 1}, 'jawOpen.5+mouthClose.5': {'jawOpen': .5, 'mouthClose': .5},
              'jawOpen+funnel': {'jawOpen': 1, 'mouthFunnel': 1}, 'jawOpen+mouthClose+smile': {'jawOpen': 1, 'mouthClose': 1, 'mouthSmileLeft': 1, 'mouthSmileRight': 1},
              'smile+cheekSquint': {'mouthSmileLeft': 1, 'mouthSmileRight': 1, 'cheekSquintLeft': 1, 'cheekSquintRight': 1}})
rep = {}
for nm, kw in POSES.items():
    V = B + sum(w * D[k] for k, w in kw.items()); N, A = normals(V)
    flip = int(((N * N0).sum(1) < 0).sum()); coll = int((A < A0 * 0.02).sum())
    if flip or coll: rep[nm] = {'flipped': flip, 'collapsed': coll}
json.dump({'poses_checked': len(POSES), 'poses_with_folds': len(rep), 'folds': rep}, open(P.OUT + 'folds.json', 'w'), indent=1)
print('poses', len(POSES), 'with folds', len(rep)); print(sorted(rep.items(), key=lambda kv: -kv[1]['flipped'])[:15])
