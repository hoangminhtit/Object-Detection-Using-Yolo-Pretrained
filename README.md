# Object Detection Application

Ứng dụng phát hiện đối tượng sử dụng YOLOv8 và Flask.

## Tính năng

- **Image Detection**: Phát hiện đối tượng trong ảnh tĩnh
- **Real-time Detection**: Phát hiện đối tượng real-time với webcam
- **Web Interface**: Giao diện web thân thiện với người dùng

## Cách sử dụng

### 1. Khởi động ứng dụng

```bash
python3 app.py
```
Truy cập: http://localhost:5000

### 2. Image Detection
1. Chọn file ảnh (JPG, PNG)
2. Điều chỉnh confidence threshold
3. Click "Áp dụng" để xử lý
4. Tải xuống ảnh đã xử lý

### 3. Real-time Detection
1. Chọn "Real Time Detection" trong dropdown
2. Click "Launch Camera Application"
3. Sử dụng các phím tắt:
   - `q`: Thoát
   - `s`: Lưu frame hiện tại
   - `c`: Thay đổi confidence threshold

## Yêu cầu hệ thống

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Flask

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc project

```
├── app.py                 # Flask web application
├── run_realtime.py        # Real-time detection script
├── imageDetection.py      # Image processing functions
├── models.py             # YOLO model wrapper
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # CSS styles
│   └── scripts.js        # JavaScript functions
└── requirements.txt      # Dependencies
```
