import cv2
from ultralytics import YOLO
import os

# ===== 模型路径（自动适配）=====
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train7', 'weights', 'best.pt')

# ===== 加载模型（只加载一次）=====
model = YOLO(MODEL_PATH)

# ===== 初始化摄像头（只初始化一次）=====
cap = cv2.VideoCapture(0)

def detect_fruit(dummy=0):
    """
    Simulink 调用入口
    返回:
        1 → apple
        2 → banana
        3 → orange
        0 → 未识别
    """

    # 1️⃣ 摄像头检查
    if not cap.isOpened():
        print("Camera not opened")
        return 0

    # 2️⃣ 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        return 0

    # 3️⃣ YOLO检测
    results = model(frame)

    # 4️⃣ 没检测到
    if len(results[0].boxes) == 0:
        print("No object detected")
        return 0

    # 5️⃣ 获取类别
    cls = int(results[0].boxes.cls[0])
    names = results[0].names
    label = names[cls]

    print("Detected:", label)

    # 6️⃣ 分类映射
    if label == 'apple':
        return 1
    elif label == 'banana':
        return 2
    elif label == 'orange':
        return 3
    else:
        return 0