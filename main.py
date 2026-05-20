import cv2
from ultralytics import YOLO

# 加载轻量级的 YOLOv8n 模型 (首次运行会自动下载约6MB的模型文件)
model = YOLO('yolov8n.pt')

# 打开 Mac 的内置摄像头 (0 代表默认摄像头)
cap = cv2.VideoCapture(0)

print("正在打开摄像头，按 'q' 键退出...")

while cap.isOpened():
    # 读取摄像头的一帧画面
    success, frame = cap.read()
    if not success:
        print("无法获取摄像头画面")
        break

    # 使用 YOLO 进行推理检测
    results = model(frame, stream=True)

    # 解析结果并在画面上画框
    for r in results:
        annotated_frame = r.plot()

    # 显示带有检测框的画面
    cv2.imshow("Mac YOLO Test - PyCharm", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()