import cv2
import time
import os
import sys
import torch
import numpy as np

# 添加 YOLOv5 仓库的路径到 Python 的模块搜索路径中
sys.path.append('E:/BISHE shijuejiance/yolov5-5.0')

# 从 YOLOv5 的模型定义文件中导入所需的模型类
from models.experimental import attempt_load

# 加载预训练的模型权重
model = attempt_load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt', map_location=torch.device('cpu'))

if not os.path.exists('img'):
    os.mkdir('img')

cap = cv2.VideoCapture(0)  # 调用摄像头'0'一般是打开电脑自带摄像头,'1'是打开外部摄像头(只有一个摄像头的情况)

if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    frame_resized = cv2.resize(frame, (640, 480))
    cv2.imshow("frame", frame_resized)

    # 将捕获的帧转换为 RGB 颜色空间,确保数据是连续的,并将其转换为 PyTorch 张量
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0  # 归一化
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # 调整维度

    # 使用模型进行推理
    with torch.no_grad():
        results = model(frame_tensor)

    # 处理检测结果
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # 置信度阈值
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Detection", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # 按 'c' 键拍照并保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"img/capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"图像已保存: {filename}")

cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 销毁窗口