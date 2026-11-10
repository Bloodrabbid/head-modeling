import os
import sys
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as P
from PIL import Image
REFS = {'front': (P.REF['front'], 0.5), 'right': (P.REF['right'], 1.0), 'left': (P.REF['left'], 1.0)}
pre, out = sys.argv[1], sys.argv[2]
rows = []
for n, (p, s) in REFS.items():
    r = Image.open(p).convert('RGB'); r = r.resize((int(r.width * s), int(r.height * s)))
    o = Image.open(f'{pre}_{n}.png').convert('RGB')
    H = 330; rows.append([im.resize((int(im.width * H / im.height), H)) for im in (r, o, Image.blend(r, o, 0.5))])
W = sum(i.width for i in rows[0]); S = Image.new('RGB', (W, 330 * 3))
for k, row in enumerate(rows):
    x = 0
    for im in row: S.paste(im, (x, 330 * k)); x += im.width
S.save(out, quality=85)
