import os
"""Head vertex ids (MakeHuman) and edges of the stage-1 head, for the silhouette term of the fit (head_topo.npz)."""
import bpy, numpy as np, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
from scipy.spatial import cKDTree
bpy.ops.wm.open_mainfile(filepath=P.WORK + 'stage1.blend')
h = bpy.data.objects['Head']
S1 = json.load(open(P.WORK + 'stage1.json')); O = np.array(S1['origin_mh'])
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
Vn = np.load(P.WORK + 'Vn.npy'); cand = np.array(list(h['orig_index']))
HV = np.array([v.co[:] for v in h.data.vertices])
hu = cand[cKDTree((Vn[cand] - O) @ A.T * 0.1).query(HV)[1]]
E = np.array([e.vertices[:] for e in h.data.edges])
np.savez(P.FIT + 'head_topo.npz', hu=hu, he=E); print('verts', len(hu), 'edges', len(E))
