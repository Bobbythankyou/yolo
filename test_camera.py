import cv2

# 运行这个脚本找到正确编号
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头 {i} 可用")
        cap.release()
    else:
        print(f"摄像头 {i} 不可用")