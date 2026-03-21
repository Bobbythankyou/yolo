import cv2
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train7/weights/best.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 键或 ESC 键退出检测")

cv2.namedWindow('水果检测')

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 进行检测
    results = model(frame)

    # 在帧上绘制检测结果
    annotated_frame = results[0].plot()

    # 显示结果
    cv2.imshow('水果检测', annotated_frame)

    # 按 'q' 键或 ESC 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is ESC
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()