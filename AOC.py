import cv2
import numpy as np
import time

def start_pi_camera_v2():
    print("🟢 正在通过新版 Bookworm 官方 Picamera2 驱动初始化 OV5647...")
    
    try:
        # 1. 导入全新的官方 Picamera2 模块
        from picamera2 import Picamera2
        picam = Picamera2()
        
        # 2. 配置视频流的分辨率（640x480 最适合树莓派跑 YOLO）
        config = picam.create_video_configuration(main={"size": (640, 480)})
        picam.configure(config)
        
        # 3. 启动相机硬件
        picam.start()
        
    except Exception as e:
        print(f"❌ 初始化 Picamera2 失败，错误原因: {e}")
        print("请确认是否运行了: sudo apt install python3-picamera2")
        return

    print("🟢 摄像头已常开，正在实时抓取画面... (按 Ctrl + C 退出)")
    
    try:
        while True:
            # 4. 关键：Picamera2 提供了完美的 capture_array() 
            # 这会直接吐出一个 100% 兼容 OpenCV 的 NumPy 矩阵（BGR/RGB 格式）
            frame = picam.capture_array()
            
            if frame is None or frame.size == 0:
                print("⚠️ 未能成功读取到画面帧...")
                continue

            # ----------------------------------------------------
            #  【核心跳板】后续你的 YOLO 检测和人脸识别代码直接在这里处理 frame
            # ----------------------------------------------------
            # 如果你看到这一行持续打印，说明你的硬件和常开监听彻底通了！
            print(f"成功抓取画面帧！图像大小: {frame.shape[1]}x{frame.shape[0]}") 
            
            # 控制一下频率，防止主循环跑太快把 CPU 撑爆
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 收到退出指令，正在关闭摄像头...")
    finally:
        # 5. 关闭相机释放硬件锁，否则下次运行会报设备忙
        picam.stop()
        print("🏁 官方摄像头已安全释放。")

if __name__ == "__main__":
    start_pi_camera_v2()