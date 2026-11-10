"""Minimal MakeHuman data readers: base mesh, morph targets, proxies (.mhclo). Pure numpy.
MakeHuman space: decimetres, Y up, Z toward the viewer. to_blender() -> metres, Z up, face toward -Y."""
import os, re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np

DATA = P.MH_DATA + ''
PACK = P.MH_PACK + ''
TGT = DATA + '/targets'


def load_obj(path):
    V, UV, faces, fuv, fgrp = [], [], [], [], []
    g = ''
    with open(path, errors='replace') as f:
        for ln in f:
            if ln.startswith('v '):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('vt '):
                UV.append([float(x) for x in ln.split()[1:3]])
            elif ln.startswith('g '):
                g = ln.split(None, 1)[1].strip()
            elif ln.startswith('f '):
                vi, ti = [], []
                for tok in ln.split()[1:]:
                    p = tok.split('/')
                    vi.append(int(p[0]) - 1)
                    ti.append(int(p[1]) - 1 if len(p) > 1 and p[1] else -1)
                faces.append(vi); fuv.append(ti); fgrp.append(g)
    return np.array(V, np.float64), np.array(UV, np.float64), faces, fuv, fgrp


def load_target(path):
    idx, off = [], []
    with open(path, errors='replace') as f:
        for ln in f:
            if ln.startswith('#') or not ln.strip(): continue
            p = ln.split()
            idx.append(int(p[0])); off.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, np.int64), np.array(off, np.float64).reshape(-1, 3)


def apply_targets(V, weights):
    """weights: {relative target path without .target: weight}"""
    V = V.copy()
    for name, w in weights.items():
        if abs(w) < 1e-9: continue
        i, o = load_target(os.path.join(TGT, name + '.target'))
        if len(i): V[i] += w * o
    return V


def load_mhclo(path):
    d = {'dir': os.path.dirname(path), 'scale': {}}
    refs, wts, offs = [], [], []
    mode = None
    with open(path, errors='replace') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'): continue
            p = s.split()
            if p[0] == 'verts': mode = 'verts'; continue
            if p[0] in ('delete_verts', 'weights', 'faces'): mode = p[0]; continue
            if mode == 'verts' and re.match(r'^-?\d', p[0]):
                if len(p) == 1:
                    v = int(p[0]); refs.append([v, v, v]); wts.append([1, 0, 0]); offs.append([0, 0, 0])
                else:
                    refs.append([int(p[0]), int(p[1]), int(p[2])]); wts.append([float(x) for x in p[3:6]])
                    offs.append([float(x) for x in p[6:9]])
                continue
            if p[0] in ('x_scale', 'y_scale', 'z_scale'):
                d['scale'][p[0][0]] = (int(p[1]), int(p[2]), float(p[3]))
            elif p[0] in ('name', 'obj_file', 'material'):
                d[p[0]] = s.split(None, 1)[1]
    d['refs'] = np.array(refs, np.int64); d['wts'] = np.array(wts, np.float64); d['offs'] = np.array(offs, np.float64)
    return d


def fit_proxy(clo, V):
    sc = np.ones(3)
    for k, ax in (('x', 0), ('y', 1), ('z', 2)):
        if k in clo['scale']:
            a, b, den = clo['scale'][k]
            sc[ax] = abs(V[a, ax] - V[b, ax]) / den
    r, w = clo['refs'], clo['wts']
    return (V[r[:, 0]] * w[:, 0:1] + V[r[:, 1]] * w[:, 1:2] + V[r[:, 2]] * w[:, 2:3]) + clo['offs'] * sc


def read_mhmat(path):
    d = {}
    with open(path, errors='replace') as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 2 and not ln.startswith('#'): d[p[0]] = p[1:]
    return d


def to_blender(V):
    """MH (x, y up, z front) decimetres -> Blender (x, -z, y) metres."""
    V = np.asarray(V)
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], -1) * 0.1
