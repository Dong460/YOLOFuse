from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("/home/dongyuting/YOLOFuse/runs/train/boat-dual-depth4/weights/best.pt")

    model.predict(
        # Point to the RGB folder; YOLOFuse will automatically pair imagesIR with the same stem.
        source="ultralytics/assets/LLVIP/images/image.png",
        save=True,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        show=False,
        device="0",
        project="runs/predict",
        name="boat-dual-depth4-val",
        save_txt=False,
        save_conf=True,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        line_width=3,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
    )
