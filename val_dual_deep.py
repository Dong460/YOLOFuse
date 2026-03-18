import json
from pathlib import Path

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("/home/dongyuting/YOLOFuse/runs/train/boat-dual-depth4/weights/best.pt")
    results = model.val(
        data="/home/dongyuting/YOLOFuse/boat_dual/dataset_dual.yaml",
        ch=6,  # RGB 3-channel + depth 3-channel
        split="val",
        imgsz=640,
        batch=16,
        device="0",
        project="runs/val",
        name="boat-dual-depth4-val",
    )

    save_dir = Path(getattr(results, "save_dir", "runs/val/boat-dual-depth4-val"))
    metrics = dict(results.results_dict)
    summary_lines = [
        "Validation metrics:",
        f"precision(B): {metrics.get('metrics/precision(B)', 'N/A')}",
        f"recall(B): {metrics.get('metrics/recall(B)', 'N/A')}",
        f"mAP50(B): {metrics.get('metrics/mAP50(B)', 'N/A')}",
        f"mAP50-95(B): {metrics.get('metrics/mAP50-95(B)', 'N/A')}",
        f"fitness: {metrics.get('fitness', 'N/A')}",
        f"save_dir: {save_dir}",
    ]

    print("\n".join(summary_lines))
    (save_dir / "metrics_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (save_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
