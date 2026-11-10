#!/usr/bin/env bash
# Landmark fit of the morph weights to three reference screenshots (front, profile nose-right, side of the left side).
# Put them at refs/front.png, refs/right.png, refs/left.png (or set REF_FRONT / REF_RIGHT / REF_LEFT).
# Needs one normal build first (10_build) for the stage-1 head. Writes build/fit/w_final.json and build/fit/cams.json.
set -euo pipefail
cd "$(dirname "$0")"
F=../build/fit; mkdir -p $F
python3 fit/lm.py $F/ref "${REF_FRONT:-../refs/front.png}" "${REF_RIGHT:-../refs/right.png}" "${REF_LEFT:-../refs/left.png}"
python3 fit/masks.py; python3 -I fit/topo.py
cp config.json $F/w0.json
python3 fit/fit.py init $F/w0.json $F/cams0.json
python3 fit/fit_render.py $F/w0.json $F/cams0.json $F/it0
prev=$F/it0.json; w=$F/w0.json; c=$F/cams0.json
for i in 1 2 3; do
  NFEV=80 TIE_F=2 BND=1.4 python3 fit/fit.py solve $w $prev $c $F/w$i.json $F/cams$i.json 0.03
  python3 fit/fit_render.py $F/w$i.json $F/cams$i.json $F/it$i
  prev="$prev,$F/it$i.json"; w=$F/w$i.json; c=$F/cams$i.json
done
cp $w $F/w_final.json; cp $c $F/cams.json
echo "fitted weights: $F/w_final.json (copy its targets into config.json, then tune by eye with fit/variants.py)"
