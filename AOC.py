import cv2
import time

def start_always_on_camera():
    # 1. 打开摄像头 (0 对应 /dev/video0)
    cap = cv2.VideoCapture(0)
    
    # 2. 设置 bookworm 系统下树莓派 4 能流畅跑的分辨率
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 降低摄像头自带的缓存队列，保证获取到的是“最实时”的画面
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查硬件连接！")
        return

    print("🟢 摄像头已常开，正在实时监听画面... (按 Ctrl + C 退出)")
    
    try:
        # 3. 进入无限循环，实现“常开”
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 丢帧或未能读取到画面，正在重试...")
                time.sleep(0.1)
                continue

            # ----------------------------------------------------
            #  此处就是你的 AI 模块切入点！
            # ----------------------------------------------------
            #  1. 丢给 YOLO 检查画面里有没有人脸 (YOLO 检测)
            #  2. 如果有人脸，裁剪出来丢给 MobileFaceNet (识别)
            #  3. 比对成功，调用 logger.py 记入 CSV
            # ----------------------------------------------------

            # 为了防止这个死循环把树莓派的 CPU 单核直接占满到 100%，
            # 必须加上一个极其微小的休眠，让 CPU 喘口气（大约每秒处理 30 帧）
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n🛑 收到退出指令，正在关闭摄像头...")
    finally:
        # 4. 无论如何，最后都要释放摄像头资源，否则下次运行会提示设备被占用
        cap.release()
        print("🏁 摄像头已成功关闭，释放资源。")

if __name__ == "__main__":
    start_always_on_camera()