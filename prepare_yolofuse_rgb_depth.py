#!/usr/bin/env python3
"""
Prepare a YOLO-format RGB dataset for YOLOFuse-style dual-input training.

What it does
------------
1. Reads a standard YOLO dataset, e.g.
   dataset/
     train/images/*.jpg
     train/labels/*.txt
     valid/images/*.jpg
     valid/labels/*.txt
     test/images/*.jpg
     test/labels/*.txt

2. Creates YOLOFuse-style layout:
   output/
     images/train/*.jpg      # RGB
     imagesIR/train/*.png    # depth-as-second-modality
     labels/train/*.txt
     images/val/*.jpg
     imagesIR/val/*.png
     labels/val/*.txt

3. Depth modality can come from either:
   - existing depth files in a mirrored folder structure, or
   - pseudo-depth generated with Depth Anything V2 via transformers.

Notes
-----
- YOLOFuse is published for RGB+IR and matches pairs by filename.
  This script reuses that convention and stores depth maps under imagesIR/
  so you can train with minimal code changes.
- Because many RGB+IR loaders expect 3-channel images, the depth map is saved
  by default as 3-channel grayscale PNG (same value repeated on RGB channels).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def find_split_layout(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """Support both:
    A) train/images + train/labels
    B) images/train + labels/train
    """
    split = split.lower()
    if split not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported split: {split}")

    # Layout A
    a_img = dataset_root / split / "images"
    a_lab = dataset_root / split / "labels"
    if a_img.exists() and a_lab.exists():
        return a_img, a_lab

    # Layout B
    b_key = split
    if split == "valid":
        b_key = "val"
    b_img = dataset_root / "images" / b_key
    b_lab = dataset_root / "labels" / b_key
    if b_img.exists() and b_lab.exists():
        return b_img, b_lab

    raise FileNotFoundError(
        f"Could not find image/label folders for split '{split}' under {dataset_root}"
    )


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def copy_labels(src_label_dir: Path, dst_label_dir: Path, image_paths: Iterable[Path]) -> int:
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_path in image_paths:
        label_src = src_label_dir / f"{img_path.stem}.txt"
        if not label_src.exists():
            print(f"[WARN] Missing label for {img_path.name}: {label_src}")
            continue
        shutil.copy2(label_src, dst_label_dir / label_src.name)
        copied += 1
    return copied


def save_depth_png(depth_arr: np.ndarray, out_path: Path, make_3ch: bool = True) -> None:
    depth_arr = np.asarray(depth_arr)
    depth_arr = np.nan_to_num(depth_arr, nan=0.0, posinf=0.0, neginf=0.0)
    dmin, dmax = float(depth_arr.min()), float(depth_arr.max())
    if dmax <= dmin:
        norm = np.zeros_like(depth_arr, dtype=np.uint8)
    else:
        norm = ((depth_arr - dmin) / (dmax - dmin) * 255.0).clip(0, 255).astype(np.uint8)

    if make_3ch:
        rgb = np.stack([norm, norm, norm], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(out_path)
    else:
        Image.fromarray(norm, mode="L").save(out_path)


def load_existing_depth(depth_file: Path) -> np.ndarray:
    img = Image.open(depth_file)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


class DepthEstimator:
    def __init__(self, model_name: str, device: str = "auto"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif device == "cpu":
            device = "cpu"
        elif device == "cuda":
            device = "cuda"
        else:
            raise ValueError("device must be one of: auto, cpu, cuda")

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            depth = outputs.predicted_depth
            depth = self.torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = depth.detach().cpu().numpy()
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float32)


def resolve_existing_depth_path(
    rgb_img: Path,
    rgb_img_dir: Path,
    depth_root: Path,
) -> Optional[Path]:
    rel = rgb_img.relative_to(rgb_img_dir)
    # 1) mirrored same relative path
    cand = depth_root / rel
    if cand.exists():
        return cand
    # 2) same stem, common depth suffixes
    for ext in [".png", ".jpg", ".jpeg", ".npy"]:
        cand = depth_root / f"{rgb_img.stem}{ext}"
        if cand.exists():
            return cand
    # 3) mirrored stem under root
    for ext in [".png", ".jpg", ".jpeg"]:
        cand = depth_root / rel.parent / f"{rgb_img.stem}{ext}"
        if cand.exists():
            return cand
    return None


def process_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    estimator: Optional[DepthEstimator],
    existing_depth_root: Optional[Path],
    copy_rgb: bool,
    make_3ch_depth: bool,
) -> Dict[str, int]:
    rgb_dir, label_dir = find_split_layout(dataset_root, split)
    out_split = SPLIT_ALIASES[split]
    out_rgb = output_root / "images" / out_split
    out_depth = output_root / "imagesIR" / out_split
    out_label = output_root / "labels" / out_split
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    images = list_images(rgb_dir)
    label_count = copy_labels(label_dir, out_label, images)
    depth_count = 0
    rgb_count = 0

    for img_path in images:
        dst_rgb = out_rgb / img_path.name
        if copy_rgb:
            shutil.copy2(img_path, dst_rgb)
        else:
            Image.open(img_path).save(dst_rgb)
        rgb_count += 1

        dst_depth = out_depth / f"{img_path.stem}.png"
        depth_arr = None

        if existing_depth_root is not None:
            depth_path = resolve_existing_depth_path(img_path, rgb_dir, existing_depth_root)
            if depth_path is None:
                print(f"[WARN] No existing depth found for {img_path.name}")
            else:
                if depth_path.suffix.lower() == ".npy":
                    depth_arr = np.load(depth_path).astype(np.float32)
                else:
                    depth_arr = load_existing_depth(depth_path)

        if depth_arr is None:
            if estimator is None:
                raise RuntimeError(
                    f"No depth source for {img_path}. Provide --existing-depth-root or enable --estimate-depth"
                )
            depth_arr = estimator.predict(img_path)

        save_depth_png(depth_arr, dst_depth, make_3ch=make_3ch_depth)
        depth_count += 1

    return {
        "rgb": rgb_count,
        "depth": depth_count,
        "labels": label_count,
        "split": out_split,
    }


def write_dataset_yaml(output_root: Path, include_test: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        yaml_lines.append("test: images/test")
    yaml_lines.extend(
        [
            "# second modality folder expected by YOLOFuse",
            "images_ir_train: imagesIR/train",
            "images_ir_val: imagesIR/val",
        ]
    )
    if include_test:
        yaml_lines.append("images_ir_test: imagesIR/test")
    yaml_lines.extend(
        [
            "",
            "# TODO: edit class names below",
            "names:",
            "  0: boat",
        ]
    )
    yaml_text = "\n".join(yaml_lines) + "\n"
    (output_root / "dataset_dual.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--existing-depth-root", type=Path, default=None)
    parser.add_argument("--estimate-depth", action="store_true")
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="transformers depth-estimation model name",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--single-channel-depth", action="store_true")
    parser.add_argument("--no-copy-rgb", action="store_true")
    args = parser.parse_args()

    if args.existing_depth_root is None and not args.estimate_depth:
        raise SystemExit("You must provide either --existing-depth-root or --estimate-depth")

    estimator = None
    if args.estimate_depth:
        estimator = DepthEstimator(args.depth_model, device=args.device)

    summaries = []
    for split in args.splits:
        try:
            summary = process_split(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                split=split,
                estimator=estimator,
                existing_depth_root=args.existing_depth_root,
                copy_rgb=not args.no_copy_rgb,
                make_3ch_depth=not args.single_channel_depth,
            )
            summaries.append(summary)
            print(
                f"[OK] {summary['split']}: rgb={summary['rgb']} depth={summary['depth']} labels={summary['labels']}"
            )
        except FileNotFoundError:
            print(f"[SKIP] split not found: {split}")

    if not summaries:
        raise SystemExit(
            "No valid dataset splits were found. "
            "Expected either <dataset-root>/<split>/images + labels "
            "or <dataset-root>/images/<split> + labels/<split>."
        )

    include_test = any(s["split"] == "test" for s in summaries)
    write_dataset_yaml(args.output_root, include_test=include_test)
    print(f"[DONE] Wrote dataset yaml: {args.output_root / 'dataset_dual.yaml'}")


if __name__ == "__main__":
    main()
