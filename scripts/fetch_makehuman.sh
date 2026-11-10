#!/usr/bin/env bash
# Downloads the MakeHuman base mesh + morph targets (GitHub) and the official CC0 system asset pack
# (skins, hair, eyes, brows, lashes, teeth, tongue) into ./vendor. About 300 MB.
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
echo "ok: vendor/makehuman/makehuman/data and vendor/pack"
