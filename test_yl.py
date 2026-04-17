from ultralytics import YOLO


def main() -> None:
    # Đường dẫn weight sau khi train xong
    model = YOLO("runs_yolo11/helmet_violation/weights/best.pt")

    # Đánh giá trên tập val
    metrics = model.val(data="data_helmet_violation.yaml")
    print(metrics)

    # Test nhanh dự đoán trên thư mục ảnh
    model.predict(
        source="D:/KHOA_HOC_MAY_TINH/HK2-2025-2026/Nhap_Mon_Thi_Giac_MT/DO_AN/dataset/helmet_violation_dataset/images/val",
        conf=0.25,
        save=True,
        project="runs_yolo11",
        name="predict_val",
    )


if __name__ == "__main__":
    main()
