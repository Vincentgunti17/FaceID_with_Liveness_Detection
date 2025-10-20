#!/bin/zsh
cd "/Users/vincentgunti/Desktop/FaceID_Practicum"
source .venv-faceid/bin/activate

# 1️⃣ Create new session name
SESSION_NAME="session_$(date +%Y%m%d_%H%M%S)"
SESSION_PATH="data/captured/vincent/$SESSION_NAME"
mkdir -p "$SESSION_PATH"

echo "\n[INFO] 🧍 Capturing new live images for session: $SESSION_NAME"
python capture_camera.py --person vincent

# 2️⃣ Automatically run verification using latest session
echo "\n[INFO] 🔍 Running FaceID verification..."
python verify_captured_vs_gallery.py \
  --gallery "data/gallery/vincent" \
  --captures "$SESSION_PATH" \
  --outdir "results/captured_vs_gallery" \
  --threshold 0.70

# 3️⃣ Done
echo "\n✅ [INFO] FaceID pipeline complete!"
echo "[INFO] Captures saved under: $SESSION_PATH"
echo "[INFO] Results saved under: results/captured_vs_gallery/"
