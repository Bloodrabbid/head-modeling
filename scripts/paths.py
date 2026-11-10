"""Where the pipeline reads and writes. Every path can be overridden with an environment variable."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.path.abspath(os.environ.get('HEAD_ROOT', os.path.join(REPO, 'build')))
WORK = os.path.join(ROOT, 'work') + os.sep                     # intermediate .blend / .json / .npy
OUT = os.path.join(ROOT, 'out') + os.sep                       # GLB, FBX, .blend, textures, previews, checks
FIT = os.path.join(ROOT, 'fit') + os.sep                       # landmark fitting: renders, correspondences, masks
MH_DATA = os.path.abspath(os.environ.get('MH_DATA', os.path.join(REPO, 'vendor', 'makehuman', 'makehuman', 'data')))
MH_PACK = os.path.abspath(os.environ.get('MH_PACK', os.path.join(REPO, 'vendor', 'pack')))
MP_MODEL = os.path.abspath(os.environ.get('MP_MODEL', os.path.join(REPO, 'vendor', 'face_landmarker.task')))
CONFIG = os.path.abspath(os.environ.get('HEAD_CONFIG', os.path.join(HERE, 'config.json')))
# reference screenshots: front, profile with the nose to the right (character's right side), side view of the left side
REF = {'front': os.environ.get('REF_FRONT', os.path.join(REPO, 'refs', 'front.png')),
       'right': os.environ.get('REF_RIGHT', os.path.join(REPO, 'refs', 'right.png')),
       'left': os.environ.get('REF_LEFT', os.path.join(REPO, 'refs', 'left.png'))}
for d in (WORK, OUT, FIT): os.makedirs(d, exist_ok=True)
