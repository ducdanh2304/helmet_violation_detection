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

    # --- AUGMENTATION ---
    
    # Lật ảnh — camera trên cao, xe đi mọi hướng
    fliplr=0.5,      # lật ngang 50%
    flipud=0.3,      # lật dọc 30% (hợp lý vì fisheye)

    # Xoay ảnh — xe máy đi theo mọi góc trong fisheye
    degrees=15,      # xoay ±15 độ

    # Scale — xe gần/xa camera có kích thước khác nhau
    scale=0.5,       # zoom in/out ±50%

    # Màu sắc — thay đổi ánh sáng ngày/đêm, nắng/mưa
    hsv_h=0.015,     # hue
    hsv_s=0.7,       # saturation
    hsv_v=0.4,       # brightness — quan trọng vì camera ngoài trời

    # Mosaic — ghép 4 ảnh, giúp detect vật thể nhỏ tốt hơn
    mosaic=1.0,      # bật 100%

    # Che khuất ngẫu nhiên — mô phỏng xe bị che bởi xe khác
    erasing=0.4,     # xóa ngẫu nhiên 1 vùng nhỏ

    # Perspective — mô phỏng biến dạng fisheye ở rìa ảnh
    perspective=0.0002,
)


if __name__ == "__main__":
    main()
