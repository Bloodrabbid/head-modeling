#!/usr/bin/env bash
# Full rebuild: shape -> textures -> LODs and export -> checks -> wireframe sheet. Results in ./build/out.
set -euo pipefail
cd "$(dirname "$0")"
for s in 10_build 20_textures 30_finish 40_verify 50_wire; do
  echo "== $s"; python3 -I "$s.py"
done
