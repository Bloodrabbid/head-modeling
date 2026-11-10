#!/usr/bin/env bash
# MakeHuman base mesh + morph targets (GitHub), the official CC0 asset pack, and the MediaPipe face landmark model, into ./vendor.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor && cd vendor
if [ ! -d makehuman ]; then
  git clone --filter=blob:none --sparse --depth 1 https://github.com/makehumancommunity/makehuman.git
  (cd makehuman && git sparse-checkout set makehuman/data/3dobjs makehuman/data/targets makehuman/data/eyes)
fi
if [ ! -d pack ]; then
  curl -fL -o sys.zip https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip
  mkdir pack && (cd pack && unzip -q ../sys.zip) && rm sys.zip
fi
[ -f face_landmarker.task ] || curl -fL -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "ok: vendor/makehuman, vendor/pack, vendor/face_landmarker.task"
