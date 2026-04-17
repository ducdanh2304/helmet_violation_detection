from ultralytics import YOLO


def main() -> None:
    # Chọn model YOLO11 để finetune
    model = YOLO("yolo11n.pt")

    # Train
    model.train(
        data="data_helmet_violation.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs_yolo11",
        name="helmet_violation",
    )


if __name__ == "__main__":
    main()
