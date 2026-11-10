import os
"""MediaPipe face landmarks (478 points) for one or more images -> json + overlay."""
import sys, json, numpy as np, cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as P
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=P.MP_MODEL),
                                    num_faces=1, output_face_blendshapes=True, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
det = vision.FaceLandmarker.create_from_options(opts)
out = {}
for path in sys.argv[2:]:
    img = cv2.imread(path); h, w = img.shape[:2]
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        print(path, 'no face'); out[os.path.abspath(path)] = None; continue
    L = np.array([[p.x * w, p.y * h, p.z * w] for p in r.face_landmarks[0]])
    bs = {b.category_name: round(b.score, 3) for b in r.face_blendshapes[0]} if r.face_blendshapes else {}
    out[os.path.abspath(path)] = {'size': [w, h], 'pts': L.round(2).tolist(), 'blend': bs}
    ov = img.copy()
    for x, y, _ in L: cv2.circle(ov, (int(x), int(y)), max(1, w // 600), (0, 255, 0), -1)
    cv2.imwrite(path.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '_lm.jpg' if False else sys.argv[1] + '_' + str(len(out)) + '.jpg', ov)
    print(path, 'ok', L[:, 0].min().round(), L[:, 0].max().round(), 'blend top', sorted(bs.items(), key=lambda t: -t[1])[:4])
json.dump(out, open(sys.argv[1] + '.json', 'w'))
