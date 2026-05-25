import cv2
import numpy as np
import time
from picamera import PiCamera # 如果报错，请尝试从 picamera2 导入：from picamera2 import Picamera2

def start_pi_camera():
    print("🟢 正在通过官方 Picamera 驱动初始化 OV5647 摄像头...")
    
    # 1. 初始化官方新架构相机
    try:
        # 新版 Bookworm 推荐写法
        from picamera2 import Picamera2
        picam = Picamera2()
        # 配置分辨率
        picam.configure(picam.create_video_configuration(main={"size": (640, 480)}))
        picam.start()
    except ImportError:
        # 如果是老版封装
        import cv2
        print("尝试进入兼容捕获模式...")
        # 如果无法使用 picamera2 库，可让 Claude 协助你配置标准的封装层
        return

    print("🟢 摄像头已常开，正在实时监听画面... (按 Ctrl + C 退出)")
    
    try:
        while True:
            # 2. 直接从官方驱动抓取一帧 NumPy 图像（完美契合 OpenCV 格式）
            frame = picam.capture_array()
            
            # 因为 NoIR 摄像头画面是红色的，这里可以顺手把它转成黑白（灰度图）
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if frame is None or frame.size == 0:
                print("⚠️ 未能读取到画面...")
                continue

            # ----------------------------------------------------
            #  【核心跳板】后续你的 YOLO 和人脸识别代码直接在这里处理 frame
            # ----------------------------------------------------
            # print("成功抓取到一帧，形状为:", frame.shape) 
            
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n🛑 收到退出指令，正在关闭摄像头...")
    finally:
        picam.stop()
        print("🏁 官方摄像头已成功关闭，释放资源。")

if __name__ == "__main__":
    start_pi_camera()