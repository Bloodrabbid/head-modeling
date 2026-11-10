import os
"""Skin masks of the reference screenshots (face silhouettes for the fit): warm skin vs grey-green background and dark hair."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
REFS = {'front': (P.REF['front'], 0.5), 'right': (P.REF['right'], 1.0),
        'left': (P.REF['left'], 1.0)}
for n, (p, s) in REFS.items():
    a = np.asarray(Image.open(p).convert('RGB')).astype(int)
    if s != 1.0: a = a[::int(round(1 / s)), ::int(round(1 / s))]
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R - B > 28) & (R > 85) & (R - G > 8)
    m = ndi.binary_opening(m, iterations=2); m = ndi.binary_closing(m, iterations=3)
    lab, nl = ndi.label(m); sizes = ndi.sum(m, lab, range(1, nl + 1)); m = ndi.binary_fill_holes(lab == (1 + np.argmax(sizes)))
    np.save(P.FIT + f'mask_{n}.npy', m); print(n, m.shape, round(float(m.mean()), 3))
