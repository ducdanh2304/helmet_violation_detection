# YOLO11 Train/Test Source

Thư mục này dùng để quản lý source train và test model YOLO11.

## Cấu trúc
- `data_helmet_violation.yaml`: file dataset config (đã chỉnh theo đường dẫn máy bạn).
- `train_yolo11.py`: script train YOLO11.
- `test_yolo11.py`: script đánh giá + predict.
- `requirements.txt`: thư viện cần cài.

## Cài thư viện
```bash
pip install -r requirements.txt
```

## Train
```bash
python train_yolo11.py
```

## Test / Validate
```bash
python test_yolo11.py
```

> Nếu đổi vị trí dataset hoặc up sang máy khác, chỉ cần sửa lại `path` trong `data_helmet_violation.yaml`.
