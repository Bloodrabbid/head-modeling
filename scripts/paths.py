"""Where the pipeline reads and writes. Every path can be overridden with an environment variable."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.path.abspath(os.environ.get('HEAD_ROOT', os.path.join(REPO, 'build')))          # results go here
WORK = os.path.join(ROOT, 'work') + os.sep                                                 # intermediate .blend / .json
OUT = os.path.join(ROOT, 'out') + os.sep                                                   # GLB, FBX, .blend, textures, previews
MH_DATA = os.path.abspath(os.environ.get('MH_DATA', os.path.join(REPO, 'vendor', 'makehuman', 'makehuman', 'data')))
MH_PACK = os.path.abspath(os.environ.get('MH_PACK', os.path.join(REPO, 'vendor', 'pack')))
CONFIG = os.path.abspath(os.environ.get('HEAD_CONFIG', os.path.join(HERE, 'config.json')))
for d in (WORK, OUT): os.makedirs(d, exist_ok=True)
