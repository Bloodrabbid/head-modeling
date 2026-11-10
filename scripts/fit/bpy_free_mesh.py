import os
"""Head vertex ids (MakeHuman) and edges, read once from the stage-1 head via its saved index map (no bpy needed)."""
import json, numpy as np, os
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as P
CACHE = P.FIT + 'head_topo.npz'
def head_vertices_edges():
    d = np.load(CACHE); return d['hu'], d['he']
