#!/usr/bin/env python3
"""Video Style Transfer Evaluation

Evaluates stylized video against original structure frames and style reference image.

Metrics:
    - CLIP Video consistency (temporal coherence via CLIP frame similarity)
    - Pixel MSE (pixel-level warped error using GMFlow optical flow)
    - FID (Frechet Inception Distance, Art-Inception backbone)
    - LPIPS (learned perceptual similarity)
    - ArtFID = (LPIPS + 1) * (FID + 1)

Usage:
    # Single video+style pair
    python3 evaluate.py --struct /path/to/original/frames \
                         --style  /path/to/style.jpg \
                         --generated /path/to/stylized/frames \
                         --frames 10

    # Batch mode: multiple pairs from txt files
    python3 evaluate.py --struct-list structs.txt \
                         --style-list  styles.txt \
                         --generated-list generated.txt \
                         --frames 10
"""

import os
import json
import glob
import argparse
import torch
import clip
import inception

from video_metric.deps.gmflow.gmflow.gmflow import GMFlow
from video_metric.pixel_mse import calculate_pixle_mse
from video_metric.frame_acc_tem_con import folder_consistency_clip
from eval_artfid import compute_art_fid


# ------------------------------------------------------------------
#  Hardcoded model paths (consistent across this project)
# ------------------------------------------------------------------
PROJECT_ROOT = "/root/paddlejob/workspace/output/0915_detection/WACV_paper/cross_image_attention"
PRETRAINED_ROOT = os.path.join(PROJECT_ROOT, "pretrained_models")
FLOW_CKPT = os.path.join(PRETRAINED_ROOT, "flow", "gmflow_sintel-0c07dcb3.pth")
ART_INCEPTION_CKPT = os.path.join(PRETRAINED_ROOT, "art_fid", "art_inception.pth")


# ------------------------------------------------------------------
#  Model loading (singleton-style, loaded once at module level)
# ------------------------------------------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP (ViT-B/32, ~350MB, auto-cached at ~/.cache/clip/)
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)

# GMFlow (optical flow, ~200MB)
_flow_ckpt = torch.load(FLOW_CKPT, map_location=lambda s, l: s)
_flow_weights = _flow_ckpt.get("model", _flow_ckpt)
_flow_model = GMFlow(
    feature_channels=128, num_scales=1, upsample_factor=8,
    num_head=1, attention_type='swin', ffn_dim_expansion=4,
    num_transformer_layers=6,
).to("cuda")
_flow_model.load_state_dict(_flow_weights, strict=False)
_flow_model.eval()

# Art-Inception (FID + ArtFID, ~80MB)
_art_ckpt = torch.load(ART_INCEPTION_CKPT, map_location=_device)
_art_model = inception.Inception3().to(_device)
_art_model.load_state_dict(_art_ckpt, strict=False)
_art_model.eval()


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
def count_images(path):
    if not os.path.isdir(path):
        return 0
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp')
    return sum(len(glob.glob(os.path.join(path, '**', e), recursive=True)) for e in exts)


# ------------------------------------------------------------------
#  Core evaluation function
# ------------------------------------------------------------------
def evaluate_one(style_path, struct_path, generated_path, n_frames=10):
    """Evaluate one style-video pair. Returns dict of metrics."""
    n_frames = min(count_images(generated_path), n_frames)
    if n_frames == 0:
        return None

    clip_video = folder_consistency_clip(generated_path, _clip_model, _clip_preprocess)
    pixel_mse = calculate_pixle_mse(struct_path, generated_path, n_frames, _flow_model)
    artfid, fid, lpips, lpips_gray = compute_art_fid(
        generated_path, style_path, struct_path, n_frames, _art_model)

    return {
        "clip_video": clip_video,
        "pixel_mse": pixel_mse,
        "fid": fid,
        "lpips": lpips,
        "artfid": artfid,
        "lpips_gray": lpips_gray,
        "nframes": n_frames,
    }


def evaluate_batch(generated_lines, struct_lines, style_lines, n_frames=10, output_txt=None):
    """Evaluate multiple style-video entries. Writes per-video metrics and aggregated CSV."""
    results = {}
    aggregate = {k: 0.0 for k in ("clip_video", "pixel_mse", "fid", "lpips", "artfid", "lpips_gray")}
    total_frames = 0

    for idx, line in enumerate(generated_lines):
        line = line.strip()
        if not line:
            continue
        generated_path = os.path.dirname(line) if line.endswith(".mp4") else line

        # Resolve style and struct paths
        if style_lines and struct_lines:
            # From parallel lists
            style_path = style_lines[idx % len(style_lines)].strip()
            struct_path = struct_lines[idx // len(style_lines)].strip()
        else:
            # Deduct from generated path: .../{struct_name}/{style_name}/...
            parts = generated_path.rstrip("/").split("/")
            style_name = parts[-2]
            struct_name = parts[-3]
            style_path = _find_style_image(struct_name, style_name)
            struct_path = _find_struct_path(struct_name)

        metrics = evaluate_one(style_path, struct_path, generated_path, n_frames)
        if metrics is None:
            continue

        results[generated_path] = metrics
        nf = metrics["nframes"]
        for k in aggregate:
            aggregate[k] += metrics[k] * nf
        total_frames += nf

        print(f"[{idx}] {generated_path}")
        print(f"     CLIP:{metrics['clip_video']:.4f} MSE:{metrics['pixel_mse']:.4f} "
              f"FID:{metrics['fid']:.4f} LPIPS:{metrics['lpips']:.4f} "
              f"ArtFID:{metrics['artfid']:.4f} LPIPS_gray:{metrics['lpips_gray']:.4f}")

        # Save per-video metric to {generated_dir}/metric.txt
        metric_file = os.path.join(os.path.dirname(generated_path), "metric.txt")
        with open(metric_file, 'w') as f:
            f.write(f"clip_image: {metrics['clip_video']}, "
                    f"pixel_mse: {metrics['pixel_mse']}, "
                    f"fid: {metrics['fid']}, "
                    f"lpips: {metrics['lpips']}, "
                    f"artfid: {metrics['artfid']}, "
                    f"lpips_gray: {metrics['lpips_gray']}\n")

    # Aggregated weighted average + output CSV
    if total_frames > 0:
        avg = {k: v / total_frames for k, v in aggregate.items()}
    else:
        avg = {k: 0.0 for k in aggregate}

    print("\n=== Weighted Averages ===")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    print(f"  total frames: {total_frames}")

    if output_txt:
        header = "Clip_Video,Pixel_MSE,FID,LPIPS,ArtFID,LPIPS_Gray,Frames\n"
        row = f"{avg['clip_video']:.4f},{avg['pixel_mse']:.4f},{avg['fid']:.4f},{avg['lpips']:.4f},{avg['artfid']:.4f},{avg['lpips_gray']:.4f},{total_frames}\n"
        mode = 'a' if os.path.exists(output_txt) else 'w'
        with open(output_txt, mode) as f:
            if mode == 'w':
                f.write("Method," + header)
            f.write(f"{os.path.basename(output_txt).replace('.txt','')}," + row)
        print(f"\nResults saved to: {output_txt}")

    return results, avg


def _find_style_image(struct_name, style_name):
    """Try to locate style image in common directories."""
    candidates = [
        f"/root/paddlejob/workspace/output/0915_detection/dataset/style_dataset/ref2sketch_yr/ref/{style_name}",
    ]
    for c in candidates:
        matches = glob.glob(f"{c}.*")
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Style image not found for {style_name}")


def _find_struct_path(struct_name):
    """Locate original video frames directory."""
    candidates = [
        f"/root/paddlejob/workspace/output/0915_detection/dataset/DAVIS/dataset/{struct_name}/imgs_crop_fore",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Struct path not found for {struct_name}")


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Style Transfer Evaluation")
    parser.add_argument("--struct", type=str, default=None,
                        help="Path to original video frames directory")
    parser.add_argument("--style", type=str, default=None,
                        help="Path to style reference image")
    parser.add_argument("--generated", type=str, default=None,
                        help="Path to generated/stylized frames directory or .mp4")
    parser.add_argument("--struct-list", type=str, default=None,
                        help="TXT file listing struct paths (one per line, batch mode)")
    parser.add_argument("--style-list", type=str, default=None,
                        help="TXT file listing style paths (one per line, batch mode)")
    parser.add_argument("--generated-list", type=str, default=None,
                        help="TXT file listing generated paths (one per line, batch mode)")
    parser.add_argument("--frames", type=int, default=10,
                        help="Max frames to evaluate per video (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output TXT file for aggregated results CSV")
    args = parser.parse_args()

    # Single pair mode
    if args.struct and args.style and args.generated:
        gen_path = os.path.dirname(args.generated) if args.generated.endswith(".mp4") else args.generated
        metrics = evaluate_one(args.style, args.struct, gen_path, args.frames)
        if metrics:
            print(f"\nCLIP: {metrics['clip_video']:.4f}  "
                  f"MSE: {metrics['pixel_mse']:.4f}  "
                  f"FID: {metrics['fid']:.4f}  "
                  f"LPIPS: {metrics['lpips']:.4f}  "
                  f"ArtFID: {metrics['artfid']:.4f}  "
                  f"LPIPS_gray: {metrics['lpips_gray']:.4f}  "
                  f"frames: {metrics['nframes']}")
        else:
            print("ERROR: No images found.")
    # Batch mode
    elif args.generated_list:
        with open(args.generated_list) as f:
            gen_lines = [l.strip() for l in f if l.strip()]
        struct_lines = []
        style_lines = []
        if args.struct_list:
            with open(args.struct_list) as f:
                struct_lines = [l.strip() for l in f if l.strip()]
        if args.style_list:
            with open(args.style_list) as f:
                style_lines = [l.strip() for l in f if l.strip()]
        evaluate_batch(gen_lines, struct_lines, style_lines, args.frames, args.output)
    else:
        parser.print_help()