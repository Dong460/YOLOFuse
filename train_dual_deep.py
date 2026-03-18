from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/fuse/Easy-level-Feature-Fusion.yaml")
    model.train(
        data="/home/dongyuting/YOLOFuse/boat_dual/dataset_dual.yaml",
        ch=6,  # RGB 3-channel + depth 3-channel
        imgsz=640,
        epochs=100,
        batch=16,
        close_mosaic=0,
        workers=8,
        device="0",
        optimizer="SGD",
        patience=0,
        amp=False,
        cache=True,
        project="runs/train",
        name="boat-dual-depth",
        resume=False,
        fraction=1.0,
    )
