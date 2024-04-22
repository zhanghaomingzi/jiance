import cv2
import time
import os
import sys
import torch
import numpy as np
from utils.general import non_max_suppression

# 添加 YOLOv5 仓库的路径到 Python 的模块搜索路径中
sys.path.append('E:/BISHE shijuejiance/yolov5-5.0')

# 从 YOLOv5 的模型定义文件中导入所需的模型类
from models.yolo import Model

# 加载预训练的模型权重
model = torch.load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt')

if os.path.exists('img') == False:
    os.mkdir('img')
import cv2
import torch

# 加载YOLOv5模型
model = torch.load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt')

cap = cv2.VideoCapture(0)  # 调用摄像头‘0’一般是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）

if False == cap.isOpened():
    print(0)
else:
    print(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5472)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3648)  # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 15)  # 设置帧率
# 显示图像
while True:
    ret, frame = cap.read()
    # print(ret)  #
    ########图像不处理的情况
    frame_1 = cv2.resize(frame, (640, 512))
    cv2.imshow("frame", frame_1)

    input = cv2.waitKey(1)
    if input == ord('q'):
        break


# 进行推理
results = model(cap)

# 应用非极大值抑制
det = results.xyxy[0].cpu().numpy()
if len(det) > 0:
    results = [torch.tensor(det)]  # 将det转换为张量并放入列表中
else:
    results = []

results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

# 绘制检测结果
for det in results:
    for *xyxy, conf, cls in det:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(cap, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(cap, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示结果
cv2.imshow('YOLOv5 Detection', cap)
cv2.waitKey(0)
cv2.destroyAllWindows()