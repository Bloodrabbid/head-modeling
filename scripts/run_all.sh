#!/usr/bin/env bash
# Full build from config.json: shape -> textures -> LODs/rig/export -> checks. Results in ./build/out.
set -euo pipefail
cd "$(dirname "$0")"
for s in 10_build 20_textures 30_finish 40_verify 45_checks 46_folds 50_wire; do echo "== $s"; python3 -I "$s.py"; done
