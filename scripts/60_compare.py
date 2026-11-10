"""Reference screenshots on top, our close-ups (work/ours_close.png from 30_finish) below.
Usage: python3 60_compare.py front.png profile.png three_quarter.png"""
import os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P

refs = sys.argv[1:4]
if len(refs) != 3: sys.exit(__doc__)
H = 520
ims = [Image.open(r).convert('RGB') for r in refs]
ims = [i.resize((int(i.width * H / i.height), H)) for i in ims]
ours = Image.open(P.WORK + 'ours_close.png').convert('RGB')
W = sum(i.width for i in ims)
ours = ours.resize((W, int(ours.height * W / ours.width)))
sheet = Image.new('RGB', (W, H + ours.height), (40, 40, 40))
x = 0
for i in ims: sheet.paste(i, (x, 0)); x += i.width
sheet.paste(ours, (0, H)); sheet.save(P.OUT + 'compare_reference.png')
print('saved', P.OUT + 'compare_reference.png')
