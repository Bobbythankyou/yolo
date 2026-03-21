# yolo_simulink.py

from ultralytics import YOLO
import numpy as np
import cv2

# 🔥 只加载一次（避免每次调用都加载模型）
model = YOLO("your_model.pt")  # 改成你的权重路径

# 类别映射（你自己改）
CLASS_MAP = {
    0: 1,  # apple
    1: 2,  # banana
    2: 3   # orange
}

def detect_fruit(dummy_input=0):
    """
    Simulink 调用入口
    输出：int (fruitType)
    """

    # 👉 这里先用测试图片（你后面可以换成摄像头）
    img = cv2.imread("test.jpg")  # 放一张测试图

    results = model(img)

    if len(results[0].boxes) == 0:
        return 0  # 没检测到

    cls = int(results[0].boxes.cls[0])

    return int(CLASS_MAP.get(cls, 0))