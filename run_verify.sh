#!/bin/zsh
cd "/Users/vincentgunti/Desktop/FaceID_Practicum"
source .venv-faceid/bin/activate
LATEST=$(ls -1t data/captured/vincent | head -n1)
python verify_captured_vs_gallery.py \
  --gallery "data/gallery/vincent" \
  --captures "data/captured/vincent/$LATEST" \
  --outdir "results/captured_vs_gallery" \
  --threshold 0.70
