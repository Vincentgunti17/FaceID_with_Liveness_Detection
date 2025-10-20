#!/usr/bin/env python3
import os, glob, json, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization

# --- Face transform (used only as fallback if MTCNN fails) ---
TF = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    fixed_image_standardization,
])

def list_images(folder):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        files.extend(glob.glob(str(Path(folder)/ext)))
    return sorted(files)

@torch.no_grad()
def embed_with_align(model, img_path, device, mtcnn):
    """Detect & align face with MTCNN; fallback to simple resize if no face."""
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)  # tensor 3x160x160 or None
    if face is None:
        # fallback
        t = TF(img).unsqueeze(0).to(device)
    else:
        t = face.unsqueeze(0).to(device)
    e = model(t)
    e = torch.nn.functional.normalize(e, dim=1)
    return e.cpu().numpy().ravel()

def cosine(a, b):
    return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Verify captured images against a gallery using FaceNet (with face alignment).")
    ap.add_argument("--gallery", required=True, help="e.g., data/gallery/vincent")
    ap.add_argument("--captures", required=True, help="e.g., data/captured/vincent/session_YYYYmmdd_HHMMSS")
    ap.add_argument("--outdir", default="results/captured_vs_gallery")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    gallery_imgs = list_images(args.gallery)
    captured_imgs = list_images(args.captures)
    if len(gallery_imgs) == 0:
        print(f"[ERROR] No images found in gallery: {args.gallery}"); return
    if len(captured_imgs) == 0:
        print(f"[ERROR] No images found in captures: {args.captures}"); return
    print(f"[INFO] Gallery images: {len(gallery_imgs)} | Captured images: {len(captured_imgs)}")

    device = args.device
    model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=None if device=="cpu" else device)

    # --- Precompute gallery embeddings ---
    gal_embs = []
    for g in gallery_imgs:
        gal_embs.append(embed_with_align(model, g, device, mtcnn))
    gal_embs = np.stack(gal_embs, axis=0)  # [G,512]

    # --- For each captured image, match to best gallery ---
    rows = []
    sims_all = []
    accept_count = 0
    for c in captured_imgs:
        ce = embed_with_align(model, c, device, mtcnn)  # [512]
        # cosine to each gallery embedding
        sims = gal_embs @ ce / (np.linalg.norm(gal_embs, axis=1)*np.linalg.norm(ce) + 1e-9)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        decision = "ACCEPT" if best_sim >= args.threshold else "REJECT"
        rows.append([c, gallery_imgs[best_idx], f"{best_sim:.4f}", decision])
        sims_all.append(best_sim)
        if decision == "ACCEPT": accept_count += 1

    sims_all = np.array(sims_all, dtype=float)
    metrics = {
        "n_gallery": int(len(gallery_imgs)),
        "n_captured": int(len(captured_imgs)),
        "threshold": float(args.threshold),
        "accepts": int(accept_count),
        "accept_rate": float(accept_count/len(captured_imgs)),
        "sim_mean": float(sims_all.mean()),
        "sim_std": float(sims_all.std()),
        "sim_min": float(sims_all.min()),
        "sim_max": float(sims_all.max()),
    }

    # Save CSV
    csv_path = outdir / "match_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["captured_image","best_gallery_match","cosine_similarity","decision(threshold={:.2f})".format(args.threshold)])
        w.writerows(rows)
    print(f"[INFO] Saved results to {csv_path}")

    # Save metrics
    with open(outdir/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("[INFO] Metrics:", metrics)

    # quick preview
    print("\n[INFO] Quick summary (top 5):")
    for r in rows[:5]:
        print("  captured:", r[0], " -> best:", r[1], "| sim:", r[2], "|", r[3])

if __name__ == "__main__":
    main()
