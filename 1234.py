import cv2
import torch
import numpy as np
from utils.general import non_max_suppression
import shexiangtoudiaoyong
from MvCameraControl_class import *
import subprocess
import os


# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

import cv2

cap = cv2.VideoCapture(1)  # 调用摄像头‘0’一般是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）

if False == cap.isOpened():
    print(0)
else:
    print(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)  # 设置图像高度
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


# 读取图像
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
if ret != 0:
    print("Enum Devices fail! ret[0x%x]" % ret)
    sys.exit()

if deviceList.nDeviceNum == 0:
    print("Find No Devices!")
    sys.exit()

print("Find %d Devices!" % deviceList.nDeviceNum)

nConnectionNum = 0  # 要连接的相机索引,从0开始
cap = shexiangtoudiaoyong.creat_camera(deviceList, nConnectionNum)

# 将图像转换为YOLOv5所需的格式
cap = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
cap = np.ascontiguousarray(cap)
cap = torch.from_numpy(cap).float() / 255.0  # 归一化
cap = cap.permute(2, 0, 1).unsqueeze(0)  # 调整维度

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