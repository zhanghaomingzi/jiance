import cv2
import torch
import numpy as np
from utils.general import non_max_suppression

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 读取图像
img = cv2.imread('image.jpg')

# 将图像转换为YOLOv5所需的格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).float() / 255.0  # 归一化
img = img.permute(2, 0, 1).unsqueeze(0)  # 调整维度

# 进行推理
results = model(img)

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
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示结果
cv2.imshow('YOLOv5 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()